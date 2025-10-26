#!/usr/bin/env python3
import streamlit as st
import os, json
import folium
from streamlit.components.v1 import html

st.set_page_config(page_title="MineAI", layout="wide")
st.title("MineAI — Map QA Preview")

georef_dir = "data/georef"
detect_dir = "data/detect"

left, right = st.columns([1,1])

with left:
    st.subheader("Georeferenced Rasters (staged)")
    geos = [f for f in os.listdir(georef_dir) if f.endswith("_georef.tif")] if os.path.isdir(georef_dir) else []
    st.write(geos if geos else "No files. Run georef stage.")

with right:
    st.subheader("Detections")
    gjs = [f for f in os.listdir(detect_dir) if f.endswith("_detect.geojson")] if os.path.isdir(detect_dir) else []
    sel = st.selectbox("Choose detection layer", gjs) if gjs else None
    if sel:
        gj = json.load(open(os.path.join(detect_dir, sel)))
        m = folium.Map(location=[39.5, -79.5], zoom_start=8, control_scale=True)
        for f in gj.get("features", []):
            if f["geometry"]["type"] == "Point":
                latlon = f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0]
                folium.CircleMarker(latlon, radius=4, tooltip=f["properties"]).add_to(m)
        html(f'<iframe srcdoc="{m.get_root().render().replace(chr(34),"&quot;")}" width="100%" height="600"></iframe>', height=620)