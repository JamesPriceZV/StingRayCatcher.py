#!/usr/bin/env python3
"""
StingRayCatcher.py — Local helper app for mapping nearby cellular sites and flagging potential cell-site simulators.

IMPORTANT LIMITATION (iPhone):
- Apple does not expose cell neighbor lists or live baseband metrics to third‑party apps over USB for non‑jailbroken iPhones.
- This tool will attempt to read basic carrier/device info via libimobiledevice (ideviceinfo) if available, but cannot pull cell tower lists directly from iPhone.
- You can still use this app to visualize and analyze cell sites by importing data (CSV/JSON) from other tools, or run a built‑in demo.

Usage examples:
- Demo map:        python main.py --demo
- Import CSV:      python main.py --csv my_cells.csv
- Import JSON:     python main.py --json my_cells.json
- Set output path: python main.py --demo --out output/map.html

CSV expected headers (case-insensitive; unknowns can be blank):
  lat, lon, operator, mcc, mnc, lac, tac, cid, pci, arfcn, rsrp, rsrq, rssi, band, timestamp

JSON expected schema:
  [ { ... same keys as CSV columns ... }, ... ]

The map will color markers by operator and highlight suspected simulators.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import webbrowser
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Lazy import folium with a clear message if missing
try:
    import folium  # type: ignore
    from folium.plugins import MarkerCluster  # type: ignore
except Exception as e:  # pragma: no cover
    folium = None  # type: ignore
    MarkerCluster = None  # type: ignore


@dataclass
class CellSite:
    lat: float
    lon: float
    operator: Optional[str] = None  # e.g., AT&T, Verizon, T-Mobile, etc.
    mcc: Optional[int] = None
    mnc: Optional[int] = None
    lac: Optional[int] = None  # 2G/3G LAI LAC
    tac: Optional[int] = None  # LTE Tracking Area Code
    cid: Optional[int] = None  # Cell ID / eNB+sector
    pci: Optional[int] = None  # Physical Cell ID (LTE/NR)
    arfcn: Optional[int] = None  # EARFCN/NR-ARFCN/UMTS-UARFCN/GSM-ARFCN
    band: Optional[str] = None
    rsrp: Optional[float] = None  # dBm
    rsrq: Optional[float] = None  # dB
    rssi: Optional[float] = None  # dBm
    timestamp: Optional[str] = None

    # Classification
    suspected_simulator: bool = False
    reasons: List[str] = None  # reasons for suspicion

    def to_popup_html(self) -> str:
        d = asdict(self)
        lines = []
        for k, v in d.items():
            if v is None or k in ("reasons", "suspected_simulator"):
                continue
            lines.append(f"<b>{k}</b>: {v}")
        if self.suspected_simulator:
            lines.append(f"<b>Flagged</b>: Potential simulator")
            if self.reasons:
                lines.append("Reasons: " + "; ".join(self.reasons))
        return "<br/>".join(lines)


# Basic maps from MNC to common US operators (extend as needed)
US_OPERATOR_BY_MCC_MNC: Dict[Tuple[int, int], str] = {
    (310, 410): "AT&T",
    (310, 150): "AT&T",
    (310, 260): "T-Mobile",
    (310, 160): "T-Mobile",
    (311, 480): "Verizon",
    (311, 490): "Verizon",
    (311, 870): "US Cellular",
}

# Carrier color scheme
CARRIER_COLORS: Dict[str, str] = {
    "AT&T": "blue",
    "Verizon": "red",
    "T-Mobile": "magenta",
    "US Cellular": "green",
}
DEFAULT_COLOR = "gray"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def try_read_iphone_info() -> Dict[str, str]:
    """Best-effort attempt to read basic info from a connected iPhone via ideviceinfo.
    Returns a dict of key info (CarrierName, InternationalMobileSubscriberIdentity, ProductType, etc.).
    Requires libimobiledevice to be installed (brew install libimobiledevice)."""
    info: Dict[str, str] = {}
    exe = shutil.which("ideviceinfo")
    if not exe:
        return info
    try:
        out = subprocess.check_output([exe], stderr=subprocess.STDOUT, text=True, timeout=5)
    except Exception:
        return info
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key in {
            "ProductType",
            "ProductVersion",
            "DeviceName",
            "InternationalMobileSubscriberIdentity",
            "BasebandVersion",
            "MobileSubscriberCountryCode",
            "MobileSubscriberNetworkCode",
            "CarrierBundleInfoArray",
            "PhoneNumber",
            "SIMStatus",
            "SubscriberCarrierNetwork",
        }:
            info[key] = val
    return info


def load_cells_from_csv(path: str) -> List[CellSite]:
    sites: List[CellSite] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            def gi(name: str) -> Optional[str]:
                # get case-insensitive
                for k, v in row.items():
                    if k.lower() == name:
                        return v
                return None

            def gi_num(name: str) -> Optional[int]:
                v = gi(name)
                if v is None or v == "":
                    return None
                try:
                    return int(float(v))
                except Exception:
                    return None
            def gi_float(name: str) -> Optional[float]:
                v = gi(name)
                if v is None or v == "":
                    return None
                try:
                    return float(v)
                except Exception:
                    return None

            lat = gi_float("lat")
            lon = gi_float("lon")
            if lat is None or lon is None:
                continue
            site = CellSite(
                lat=lat,
                lon=lon,
                operator=gi("operator"),
                mcc=gi_num("mcc"),
                mnc=gi_num("mnc"),
                lac=gi_num("lac"),
                tac=gi_num("tac"),
                cid=gi_num("cid"),
                pci=gi_num("pci"),
                arfcn=gi_num("arfcn"),
                band=gi("band"),
                rsrp=gi_float("rsrp"),
                rsrq=gi_float("rsrq"),
                rssi=gi_float("rssi"),
                timestamp=gi("timestamp"),
                reasons=[],
            )
            sites.append(site)
    return sites


def load_cells_from_json(path: str) -> List[CellSite]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sites: List[CellSite] = []
    for item in data:
        def gv(name: str):
            for k, v in item.items():
                if k.lower() == name:
                    return v
            return None
        def gint(name: str) -> Optional[int]:
            v = gv(name)
            try:
                return None if v is None or v == "" else int(v)
            except Exception:
                try:
                    return int(float(v))
                except Exception:
                    return None
        def gfloat(name: str) -> Optional[float]:
            v = gv(name)
            try:
                return None if v is None or v == "" else float(v)
            except Exception:
                return None
        lat = gfloat("lat")
        lon = gfloat("lon")
        if lat is None or lon is None:
            continue
        sites.append(CellSite(
            lat=lat,
            lon=lon,
            operator=gv("operator"),
            mcc=gint("mcc"),
            mnc=gint("mnc"),
            lac=gint("lac"),
            tac=gint("tac"),
            cid=gint("cid"),
            pci=gint("pci"),
            arfcn=gint("arfcn"),
            band=str(gv("band")) if gv("band") is not None else None,
            rsrp=gfloat("rsrp"),
            rsrq=gfloat("rsrq"),
            rssi=gfloat("rssi"),
            timestamp=str(gv("timestamp")) if gv("timestamp") is not None else None,
            reasons=[],
        ))
    return sites


def normalize_operator(site: CellSite) -> None:
    # If operator missing, try infer from MCC/MNC for some known pairs
    if not site.operator and site.mcc is not None and site.mnc is not None:
        site.operator = US_OPERATOR_BY_MCC_MNC.get((site.mcc, site.mnc))


def classify_sites(sites: List[CellSite], ref_point: Optional[Tuple[float, float]] = None) -> None:
    """Mark suspected simulators based on simple heuristics.
    Heuristics (coarse, extend as needed):
      - Operator missing and MCC/MNC unknown => suspect.
      - MCC/MNC pair maps to a different operator name than provided => suspect.
      - Unusually strong signal (RSRP > -65 dBm or RSSI > -50 dBm) with very close cluster of IDs => suspect.
      - TAC/LAC/CID values in very low ranges (<= 1) => suspect.
    """
    # Normalize operators
    for s in sites:
        normalize_operator(s)
        s.reasons = s.reasons or []

    # Map of approximate clusters by location to analyze density
    # Compute centroid if ref not given
    if not ref_point and sites:
        lat_avg = sum(s.lat for s in sites) / len(sites)
        lon_avg = sum(s.lon for s in sites) / len(sites)
        ref_point = (lat_avg, lon_avg)

    # Group by operator for color coding
    for s in sites:
        # Heuristic: missing operator and MCC/MNC unknown
        if not s.operator and (s.mcc is None or s.mnc is None):
            s.suspected_simulator = True
            s.reasons.append("Missing operator and MCC/MNC")

        # Heuristic: operator mismatch vs MCC/MNC map
        if s.mcc is not None and s.mnc is not None:
            mapped = US_OPERATOR_BY_MCC_MNC.get((s.mcc, s.mnc))
            if mapped and s.operator and mapped.lower() != s.operator.lower():
                s.suspected_simulator = True
                s.reasons.append(f"MCC/MNC ({s.mcc}-{s.mnc}) mismatch: expected {mapped}")

        # Heuristic: Abnormally strong signal
        strong = (s.rsrp is not None and s.rsrp > -65) or (s.rssi is not None and s.rssi > -50)
        if strong:
            s.suspected_simulator = True
            s.reasons.append("Unusually strong signal strength")

        # Heuristic: Low codes
        if (s.tac is not None and s.tac <= 1) or (s.lac is not None and s.lac <= 1) or (s.cid is not None and s.cid <= 1):
            s.suspected_simulator = True
            s.reasons.append("Low TAC/LAC/CID values")

    # Heuristic: high density cluster within ~50 meters with multiple different PCIs for same operator -> suspect cluster
    # Build simple grid
    grid: Dict[Tuple[int, int], List[int]] = {}
    for idx, s in enumerate(sites):
        key = (int(s.lat * 200), int(s.lon * 200))  # ~0.005 deg ~ 550 m; scaled to get ~50-100 m buckets depending on latitude
        grid.setdefault(key, []).append(idx)
    for bucket, indices in grid.items():
        if len(indices) >= 4:
            # If at least 4 cells in tiny area, flag those with strongest power
            local = [sites[i] for i in indices]
            local.sort(key=lambda x: (x.rsrp or -200), reverse=True)
            for s in local[:2]:
                s.suspected_simulator = True
                s.reasons.append("Dense cluster with strong power")


def carrier_color(carrier: Optional[str]) -> str:
    if not carrier:
        return DEFAULT_COLOR
    for name, color in CARRIER_COLORS.items():
        if carrier.lower() == name.lower():
            return color
    return DEFAULT_COLOR


def plot_map(sites: List[CellSite], out_path: str, auto_open: bool = True) -> str:
    if folium is None:
        print("This command requires folium. Install with: pip install folium", file=sys.stderr)
        sys.exit(2)

    if not sites:
        raise ValueError("No sites to plot")

    center_lat = sum(s.lat for s in sites) / len(sites)
    center_lon = sum(s.lon for s in sites) / len(sites)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, control_scale=True)

    cluster = MarkerCluster().add_to(m)

    for s in sites:
        color = carrier_color(s.operator)
        icon = "info-sign"
        if s.suspected_simulator:
            color = "black"
            icon = "exclamation-sign"
        popup = folium.Popup(s.to_popup_html(), max_width=350)
        folium.Marker(
            location=[s.lat, s.lon],
            popup=popup,
            icon=folium.Icon(color=color, icon=icon),
            tooltip=(s.operator or "Unknown") + (" (SIMULATOR?)" if s.suspected_simulator else ""),
        ).add_to(cluster)

    # Add a legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 9999; background: white; padding: 10px; border: 1px solid #444;">
      <b>Legend</b><br>
      <span style="color: blue;">■</span> AT&T<br>
      <span style="color: red;">■</span> Verizon<br>
      <span style="color: magenta;">■</span> T-Mobile<br>
      <span style="color: green;">■</span> US Cellular<br>
      <span style="color: black;">■</span> Suspected Simulator
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    m.save(out_path)
    print(f"Saved map: {out_path}")
    if auto_open:
        try:
            webbrowser.open(f"file://{os.path.abspath(out_path)}")
        except Exception:
            pass
    return out_path


def demo_sites(center_lat: float = 40.7580, center_lon: float = -73.9855) -> List[CellSite]:
    random.seed(42)
    sites: List[CellSite] = []
    carriers = ["AT&T", "Verizon", "T-Mobile"]
    for i in range(12):
        dlat = random.uniform(-0.01, 0.01)
        dlon = random.uniform(-0.01, 0.01)
        op = random.choice(carriers)
        rsrp = random.uniform(-120, -70)
        sites.append(CellSite(
            lat=center_lat + dlat,
            lon=center_lon + dlon,
            operator=op,
            mcc=310,
            mnc=410 if op == "AT&T" else (260 if op == "T-Mobile" else 480),
            tac=random.randint(10, 65535),
            cid=random.randint(100, 500000),
            pci=random.randint(0, 503),
            arfcn=random.randint(500, 1000),
            band="B2",
            rsrp=rsrp,
            rsrq=random.uniform(-20, -3),
            rssi=rsrp + random.uniform(20, 40),
            timestamp=datetime.utcnow().isoformat(),
            reasons=[],
        ))
    # Add a suspected simulator cluster
    for i in range(5):
        dlat = random.uniform(0.0002, 0.0008)
        dlon = random.uniform(0.0002, 0.0008)
        sites.append(CellSite(
            lat=center_lat + dlat,
            lon=center_lon + dlon,
            operator=None,
            mcc=None,
            mnc=None,
            tac=0 if i == 0 else 1,
            cid=1 if i < 2 else random.randint(2, 10),
            pci=random.randint(0, 503),
            arfcn=900,
            band="B5",
            rsrp=random.uniform(-60, -45),  # very strong
            rsrq=random.uniform(-8, -2),
            rssi=random.uniform(-45, -30),
            timestamp=datetime.utcnow().isoformat(),
            reasons=[],
        ))
    classify_sites(sites)
    return sites


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Map cell sites and flag simulators. iPhone support is limited by iOS restrictions.")
    gsrc = p.add_mutually_exclusive_group(required=False)
    gsrc.add_argument("--demo", action="store_true", help="Run with demo data")
    gsrc.add_argument("--csv", type=str, help="Load cell data from CSV")
    gsrc.add_argument("--json", type=str, help="Load cell data from JSON")
    p.add_argument("--out", type=str, default="cell_map.html", help="Output HTML map path")
    p.add_argument("--no-open", action="store_true", help="Do not auto-open the map in a browser")
    p.add_argument("--center-lat", type=float, help="Demo: center latitude")
    p.add_argument("--center-lon", type=float, help="Demo: center longitude")
    p.add_argument("--no-iphone", action="store_true", help="Skip iPhone info probe even if tools are present")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    # Optional: probe iPhone info
    if not args.no_iphone:
        info = try_read_iphone_info()
        if info:
            print("Connected iPhone (best-effort info):")
            for k, v in info.items():
                print(f"  {k}: {v}")
            print("Note: iOS does not expose live cell tower lists to third-party tools over USB. Import data via --csv/--json.")
        else:
            print("No iPhone info available. To enable, install libimobiledevice: brew install libimobiledevice")

    # Load sites
    sites: List[CellSite] = []
    if args.csv:
        sites = load_cells_from_csv(args.csv)
    elif args.json:
        sites = load_cells_from_json(args.json)
    else:
        # default to demo if nothing provided
        if not args.demo:
            print("No input provided. Running demo. Use --csv/--json to load your own data.")
        lat = args.center_lat if args.center_lat is not None else 40.7580
        lon = args.center_lon if args.center_lon is not None else -73.9855
        sites = demo_sites(lat, lon)

    if not sites:
        print("No sites loaded. Nothing to plot.")
        return 1

    # Classify if not already
    classify_sites(sites)

    auto_open = not args.no_open
    try:
        plot_map(sites, args.out, auto_open=auto_open)
    except Exception as e:
        print(f"Failed to create map: {e}", file=sys.stderr)
        return 2

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
