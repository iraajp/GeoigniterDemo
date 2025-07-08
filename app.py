import streamlit as st
import numpy as np
import folium
from streamlit_folium import st_folium
import rasterio
from folium.raster_layers import ImageOverlay
from branca.colormap import LinearColormap

st.set_page_config(layout="wide")
st.title("🔥 GeoIgniter - Forest Fire Risk Map")
st.markdown("Predicted fire risk map with a 0–1 risk scale overlay.")

@st.cache_data
def run_prediction():
    # Load pre-trained model
    from tensorflow.keras.models import load_model
    model = load_model("models/model.keras")

    # Load preprocessed inputs
    slope = np.load("data/slope.npy")      # (H, W)
    lulc = np.load("data/lulc.npy")        # (H, W)

    # Resize to smallest common shape
    target_shape = (min(slope.shape[0], lulc.shape[0]),
                    min(slope.shape[1], lulc.shape[1]))
    slope = slope[:target_shape[0], :target_shape[1]]
    lulc = lulc[:target_shape[0], :target_shape[1]]

    # Stack and predict
    X = np.stack([slope, lulc], axis=-1)
    X_flat = X.reshape(-1, 2)
    y_pred = model.predict(X_flat, verbose=0)
    fire_map = y_pred.reshape(target_shape)

    return fire_map, target_shape

def add_html_legend(map_object):
    legend_html = '''
     <div style="position: fixed; 
                 bottom: 50px; left: 50px; width: 160px; height: 120px; 
                 background-color: white;
                 border:2px solid grey; z-index:9999; font-size:14px;
                 padding: 10px;">
     <b>🔥 Fire Risk Scale</b><br>
     <i style="background: #00ff00; width: 10px; height: 10px; display: inline-block;"></i> 0.0 (Low)<br>
     <i style="background: #ffff00; width: 10px; height: 10px; display: inline-block;"></i> 0.5 (Medium)<br>
     <i style="background: #ff0000; width: 10px; height: 10px; display: inline-block;"></i> 1.0 (High)<br>
     </div>
     '''
    map_object.get_root().html.add_child(folium.Element(legend_html))

def main():
    fire_map, shape = run_prediction()

    # Normalize to 0-1 just in case
    norm_fire_map = (fire_map - fire_map.min()) / (fire_map.max() - fire_map.min())

    m = folium.Map(location=[30.0, 78.0], zoom_start=6, tiles="cartodbpositron")

    # Create RGB image from risk
    colormap = LinearColormap(['green', 'yellow', 'red'], vmin=0, vmax=1)
    rgb_image = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            color = colormap(norm_fire_map[i, j], rgb=False)
            rgb_image[i, j, :3] = np.array(color[:3]) * 255
            rgb_image[i, j, 3] = 100  # semi-transparent alpha

    # Flip and render
    rgb_image = np.flipud(rgb_image)
    img_overlay = folium.raster_layers.ImageOverlay(
        image=rgb_image,
        bounds=[[29.0, 77.0], [31.5, 79.5]],
        opacity=0.6,
        interactive=True,
        cross_origin=False,
        zindex=1
    )
    img_overlay.add_to(m)

    # Add both legends
    colormap.caption = '🔥 Fire Risk Probability'
    colormap.add_to(m)
    add_html_legend(m)

    st_data = st_folium(m, width=900, height=600)

if __name__ == "__main__":
    main()
