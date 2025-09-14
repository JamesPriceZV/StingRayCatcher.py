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
- Web UI:          python main.py --web

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
from datetime import datetime, UTC
from typing import List, Dict, Any, Optional, Tuple

# Optional web UI dependencies
try:
    from flask import Flask, jsonify, request
except ImportError:  # pragma: no cover
    Flask = None  # type: ignore
    jsonify = None  # type: ignore
    request = None  # type: ignore

# Lazy import folium with a clear message if missing
try:
    import folium  # type: ignore
    from folium.plugins import MarkerCluster  # type: ignore
except ImportError:  # pragma: no cover
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


def _parse_row(row: Dict[str, Any]) -> Optional[CellSite]:
    """Helper to parse a dict (from CSV or JSON) into a CellSite."""
    def g(key: str, default: Any = None) -> Any:
        for k, v in row.items():
            if k.lower() == key.lower():
                return v
        return default

    def g_float(key: str) -> Optional[float]:
        v = g(key)
        if v is None or v == '':
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    def g_int(key: str) -> Optional[int]:
        v = g(key)
        if v is None or v == '':
            return None
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return None

    lat = g_float("lat")
    lon = g_float("lon")
    if lat is None or lon is None:
        return None

    return CellSite(
        lat=lat,
        lon=lon,
        operator=g("operator"),
        mcc=g_int("mcc"),
        mnc=g_int("mnc"),
        lac=g_int("lac"),
        tac=g_int("tac"),
        cid=g_int("cid"),
        pci=g_int("pci"),
        arfcn=g_int("arfcn"),
        band=g("band"),
        rsrp=g_float("rsrp"),
        rsrq=g_float("rsrq"),
        rssi=g_float("rssi"),
        timestamp=g("timestamp"),
        reasons=[],
    )


def load_cells_from_csv(path: str) -> List[CellSite]:
    sites: List[CellSite] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            site = _parse_row(row)
            if site:
                sites.append(site)
    return sites


def load_cells_from_json(path: str) -> List[CellSite]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sites: List[CellSite] = []
    for item in data:
        site = _parse_row(item)
        if site:
            sites.append(site)
    return sites


def normalize_operator(site: CellSite) -> None:
    # If operator missing, try infer from MCC/MNC for some known pairs
    if not site.operator and site.mcc is not None and site.mnc is not None:
        site.operator = US_OPERATOR_BY_MCC_MNC.get((site.mcc, site.mnc))


def classify_sites(sites: List[CellSite]) -> None:
    """Mark suspected simulators based on simple heuristics."""
    # Normalize operators
    for s in sites:
        normalize_operator(s)
        s.reasons = s.reasons or []

    # Heuristics for individual sites
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
        if strong and "Unusually strong signal strength" not in s.reasons:
            s.suspected_simulator = True
            s.reasons.append("Unusually strong signal strength")

        # Heuristic: Low codes
        if (s.tac is not None and s.tac <= 1) or (s.lac is not None and s.lac <= 1) or (s.cid is not None and s.cid <= 1):
            if "Low TAC/LAC/CID values" not in s.reasons:
                s.suspected_simulator = True
                s.reasons.append("Low TAC/LAC/CID values")

    # Heuristic: high density cluster
    grid: Dict[Tuple[int, int], List[int]] = {}
    for idx, s in enumerate(sites):
        key = (int(s.lat * 200), int(s.lon * 200))
        grid.setdefault(key, []).append(idx)
    for bucket, indices in grid.items():
        if len(indices) >= 4:
            local = [sites[i] for i in indices]
            local.sort(key=lambda x: (x.rsrp or -200), reverse=True)
            for s in local[:2]:
                if "Dense cluster with strong power" not in s.reasons:
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

    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 9999; background: white; padding: 10px; border: 1px solid #444; border-radius: 5px;">
      <b>Legend</b><br>
      <span style="color: blue;">■</span> AT&T<br>
      <span style="color: red;">■</span> Verizon<br>
      <span style="color: magenta;">■</span> T-Mobile<br>
      <span style="color: green;">■</span> US Cellular<br>
      <span style="color: gray;">■</span> Other<br>
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
    random.seed(int(datetime.now().timestamp()))
    sites: List[CellSite] = []
    carriers = ["AT&T", "Verizon", "T-Mobile"]
    for i in range(50):
        dlat = random.uniform(-0.05, 0.05)
        dlon = random.uniform(-0.05, 0.05)
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
            timestamp=datetime.now(UTC).isoformat(),
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
            timestamp=datetime.now(UTC).isoformat(),
            reasons=[],
        ))
    return sites


# ---------- Web UI state and helpers ----------
ALL_SITES: List[CellSite] = []
LAST_REFRESH: Optional[str] = None
IPHONE_INFO: Dict[str, str] = {}
DATA_SOURCE: Dict[str, Any] = {
    'mode': None,
    'path': None,
    'center': None,
}

def sites_to_json(sites: List[CellSite]) -> List[Dict[str, Any]]:
    return [asdict(s) for s in sites]


def refresh_data(source: Dict[str, Any]) -> None:
    global ALL_SITES, LAST_REFRESH, IPHONE_INFO
    try:
        IPHONE_INFO = try_read_iphone_info()
    except Exception:
        IPHONE_INFO = {}

    mode = source.get('mode')
    new_sites: List[CellSite] = []
    if mode == 'csv' and source.get('path'):
        path = str(source['path'])
        if os.path.exists(path):
            new_sites = load_cells_from_csv(path)
    elif mode == 'json' and source.get('path'):
        path = str(source['path'])
        if os.path.exists(path):
            new_sites = load_cells_from_json(path)
    else: # demo mode
        center = source.get('center') or (40.7580, -73.9855)
        lat, lon = center
        # jitter center a bit to simulate movement on refresh
        lat += random.uniform(-0.0005, 0.0005)
        lon += random.uniform(-0.0005, 0.0005)
        new_sites = demo_sites(lat, lon)
        source['center'] = (lat, lon)

    if new_sites:
        classify_sites(new_sites)
        ALL_SITES = new_sites
        LAST_REFRESH = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def build_index_html() -> str:
    # Self-contained Leaflet UI with interactive filtering and refresh
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8' />
  <title>StingRayCatcher Web UI</title>
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>
  <style>
    body {{ margin:0; font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; display: flex; flex-direction: column; height: 100vh; }}
    #topbar {{ padding: 8px 12px; background: #f5f5f7; border-bottom: 1px solid #ddd; display:flex; gap:15px; align-items:center; flex-wrap:wrap; }}
    #map {{ flex: 1; }}
    .badge {{ padding:2px 6px; border-radius:4px; background:#eee; }}
    .sim {{ background:#000; color:#fff; }}
    .control-group {{ display: flex; align-items: center; gap: 8px; }}
    #iphone-status {{ font-size: 0.9em; color: #555; }}
  </style>
</head>
<body>
  <div id='topbar'>
    <div class='control-group'>
        <button id='refreshBtn'>Refresh</button>
        <label><input type='checkbox' id='autoToggle'/>Auto</label>
        <input id='intervalInput' type='number' min='2' value='10' style='width:50px'/>s
    </div>
    <div class='control-group'>
        <label for='distSlider'>Radius:</label>
        <input type='range' id='distSlider' min='1' max='50' value='50' style='width:100px'>
        <span id='distLabel'>All</span>
    </div>
    <div id='status' style='font-size: 0.9em;'></div>
    <div style='flex:1'></div>
    <div id='iphone-status'>Checking for iPhone...</div>
  </div>
  <div id='map'></div>
  <script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>
  <script>
    let map, layer, allSites = [];
    const carrierColors = {{'AT&T':'blue','Verizon':'red','T-Mobile':'magenta','US Cellular':'green'}};
    const defaultColor = '{DEFAULT_COLOR}';

    function carrierColor(op) {{
      if (!op) return defaultColor;
      const key = typeof op === 'string' ? op.toLowerCase() : '';
      for (const k in carrierColors) {{ if (k.toLowerCase()===key) return carrierColors[k]; }}
      return defaultColor;
    }}

    function initMap() {{
      map = L.map('map').setView([40.7580, -73.9855], 13);
      L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 19, attribution: '&copy; OpenStreetMap' }}).addTo(map);
      layer = L.layerGroup().addTo(map);
      map.on('moveend', renderMarkers);
      loadData(true);
    }}

    function renderMarkers() {{
        layer.clearLayers();
        const center = map.getCenter();
        const maxDistKm = document.getElementById('distSlider').value;
        const showAll = maxDistKm >= 50;

        allSites.forEach(s => {{
            const dist = map.distance(center, [s.lat, s.lon]) / 1000;
            if (!showAll && dist > maxDistKm) return;

            const color = s.suspected_simulator ? 'black' : carrierColor(s.operator);
            const marker = L.circleMarker([s.lat, s.lon], {{ radius: 8, color: color, fillColor: color, fillOpacity: 0.8 }});
            const lines = [];
            for (const k in s) {{ if (s[k]!==null && k!=='reasons' && s[k]!=='') lines.push(`<b>${{k}}</b>: ${{s[k]}}`); }}
            if (s.suspected_simulator) {{ lines.push('<b>Flagged</b>: Potential simulator'); if (s.reasons&&s.reasons.length) lines.push('Reasons: '+s.reasons.join('; ')); }}
            marker.bindPopup(lines.join('<br/>'));
            marker.addTo(layer);
        }});
    }}

    async function loadIphone() {{
      const iphoneEl = document.getElementById('iphone-status');
      try {{
        const r = await fetch('/iphone');
        const j = await r.json();
        const parts = [];
        if (j.ProductType) parts.push(j.ProductType);
        if (j.SubscriberCarrierNetwork) parts.push(j.SubscriberCarrierNetwork);
        iphoneEl.textContent = parts.length ? '📱 ' + parts.join(' | ') : 'iPhone not found.';
      }} catch(e) {{ iphoneEl.textContent = 'iPhone check failed.'; }}
    }}

    async function loadData(fit=false) {{
      const status = document.getElementById('status');
      status.textContent = 'Loading...';
      try {{
        const r = await fetch('/data');
        const j = await r.json();
        allSites = j.sites;
        const latlngs = allSites.map(s => [s.lat, s.lon]).filter(Boolean);

        if (fit && latlngs.length) {{ map.fitBounds(latlngs, {{padding:[50,50]}}); }}
        renderMarkers();
        status.textContent = `Source: ${{j.source}} | Last refresh: ${{j.last_refresh || 'n/a'}} | Sites: ${{allSites.length}}`;
      }} catch(e) {{
        status.textContent = 'Error loading data.';
      }}
      loadIphone();
    }}

    async function doRefresh() {{
      try {{ await fetch('/refresh', {{method:'POST'}}); }} catch(e) {{}}
      await loadData();
    }}

    let timer = null;
    function updateAuto() {{
      const on = document.getElementById('autoToggle').checked;
      const iv = Math.max(2, parseInt(document.getElementById('intervalInput').value||'10',10));
      if (timer) {{ clearInterval(timer); timer = null; }}
      if (on) {{ timer = setInterval(doRefresh, iv*1000); }}
    }}

    document.addEventListener('DOMContentLoaded', () => {{
      initMap();
      document.getElementById('refreshBtn').addEventListener('click', doRefresh);
      document.getElementById('autoToggle').addEventListener('change', updateAuto);
      document.getElementById('intervalInput').addEventListener('change', updateAuto);
      const distSlider = document.getElementById('distSlider');
      const distLabel = document.getElementById('distLabel');
      distSlider.addEventListener('input', () => {{
        distLabel.textContent = distSlider.value >= 50 ? 'All' : `${{distSlider.value}} km`;
      }});
      distSlider.addEventListener('change', renderMarkers);
    }});
  </script>
</body>
</html>
"""


def start_web_server(args: argparse.Namespace) -> int:
    if Flask is None:
        print('The web UI requires Flask. Install with: pip install flask', file=sys.stderr)
        return 2
    app = Flask(__name__)

    # Initialize data source from args
    if args.csv:
        DATA_SOURCE['mode'] = 'csv'; DATA_SOURCE['path'] = args.csv
    elif args.json:
        DATA_SOURCE['mode'] = 'json'; DATA_SOURCE['path'] = args.json
    else:
        DATA_SOURCE['mode'] = 'demo'
        if args.center_lat is not None and args.center_lon is not None:
            DATA_SOURCE['center'] = (args.center_lat, args.center_lon)

    refresh_data(DATA_SOURCE)

    @app.get('/')
    def index():  # type: ignore
        return build_index_html()

    @app.get('/data')
    def data():  # type: ignore
        return jsonify({
            'sites': sites_to_json(ALL_SITES),
            'last_refresh': LAST_REFRESH,
            'source': DATA_SOURCE.get('mode'),
        })

    @app.get('/iphone')
    def iphone():  # type: ignore
        return jsonify(IPHONE_INFO)

    @app.post('/refresh')
    def refresh():  # type: ignore
        refresh_data(DATA_SOURCE)
        return jsonify({'ok': True, 'last_refresh': LAST_REFRESH})

    host = '127.0.0.1'
    port = getattr(args, 'port', 5000)
    url = f"http://{host}:{port}"
    print(f"Web UI running at {url}  (Press CTRL+C to stop)")
    try:
        webbrowser.open(url)
    except Exception:
        pass # Fail silently if browser can't be opened

    try:
        app.run(host=host, port=int(port), debug=False)
    except KeyboardInterrupt:
        pass
    return 0


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Map cell sites and flag simulators. iPhone support is limited by iOS restrictions.")
    gsrc = p.add_mutually_exclusive_group(required=False)
    gsrc.add_argument("--demo", action="store_true", help="Run with demo data")
    gsrc.add_argument("--csv", type=str, help="Load cell data from CSV")
    gsrc.add_argument("--json", type=str, help="Load cell data from JSON")
    p.add_argument("--out", type=str, default="cell_map.html", help="Output HTML map path (CLI mode)")
    p.add_argument("--no-open", action="store_true", help="Do not auto-open the map in a browser (CLI mode)")
    p.add_argument("--center-lat", type=float, help="Demo: center latitude")
    p.add_argument("--center-lon", type=float, help="Demo: center longitude")
    p.add_argument("--no-iphone", action="store_true", help="Skip iPhone info probe even if tools are present")
    p.add_argument("--web", action="store_true", help="Start local web UI (http://127.0.0.1:5000)")
    p.add_argument("--port", type=int, default=5000, help="Web UI port")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    # Web UI mode
    if getattr(args, 'web', False):
        return start_web_server(args)

    # CLI mode behavior
    if not args.no_iphone:
        info = try_read_iphone_info()
        if info:
            print("Connected iPhone (best-effort info):")
            for k, v in info.items():
                print(f"  {k}: {v}")
            print("\nNote: iOS does not expose live cell tower lists to third-party tools over USB.")
        else:
            print("No iPhone info available. To enable, install libimobiledevice: brew install libimobiledevice")

    # Load sites
    sites: List[CellSite] = []
    if args.csv:
        sites = load_cells_from_csv(args.csv)
    elif args.json:
        sites = load_cells_from_json(args.json)
    else:
        if not args.demo:
            print("No input provided. Running demo. Use --csv/--json to load your own data, or --web for the UI.")
        lat = args.center_lat if args.center_lat is not None else 40.7580
        lon = args.center_lon if args.center_lon is not None else -73.9855
        sites = demo_sites(lat, lon)

    if not sites:
        print("No sites loaded. Nothing to plot.")
        return 1

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
