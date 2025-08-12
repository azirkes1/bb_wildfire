import ee
import streamlit as st
from google.oauth2 import service_account
import json
import folium 
from folium.plugins import Draw
from streamlit_folium import st_folium
import numpy as np
import io
from rasterio import MemoryFile
from PIL import Image
import zipfile
import requests
import io
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as cx
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from PIL import ImageDraw, ImageFont
from PIL import ImageChops
from PIL import ImageOps
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from shapely.geometry import box
from pyproj import Transformer
import cartopy.crs as ccrs
import rasterio
from rasterio.crs import CRS
from PIL import ImageDraw, ImageFont
from matplotlib_scalebar.scalebar import ScaleBar
from rasterio.warp import calculate_default_transform, reproject, Resampling
import streamlit as st
import requests
import geopandas as gpd
from folium.plugins import MarkerCluster
from scipy.ndimage import generic_filter
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
# ---------------------------------------------------------
#  authorize Google Earth Engine 
# ---------------------------------------------------------
json_data = st.secrets["json_data"]
service_account = st.secrets["service_account"]

#prepare values
json_object = json.loads(json_data, strict=False)
service_account = json_object['client_email']
json_object = json.dumps(json_object)

#authorize the app
credentials = ee.ServiceAccountCredentials(service_account, key_data=json_object)
ee.Initialize(credentials)

# ---------------------------------------------------------
#  streamlit app layout
# ---------------------------------------------------------

#everything resides in this container, helps to reduce padding
with st.container():

    #html page configuration 
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        /* Reduce top and bottom padding in main layout */
        .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
        }

        /* Reduce space under folium map */
        .element-container:has(.folium-map),
        iframe {
            margin-bottom: 0px !important;
        }

        /* Control folium map height */
        .folium-map {
            height: 500px !important;
            overflow: hidden !important;
        }

        /* Hide Streamlit branding/footer */
        footer, header, .stDeployButton {
            display: none !important;
        }

        /* Customize sidebar appearance */
        section[data-testid="stSidebar"] {
            overflow: hidden !important;
            max-height: none !important;
            width: 350px !important;
        }

        section[data-testid="stSidebar"] > div {
            overflow: hidden !important;
        }
                
        /* Force iframe height and remove default margin */
        iframe {
            height: 500px !important;
            display: block;
            margin: 0 auto !important;
            padding: 0 !important;
            border: none !important;
        }

        /* Hide extra container spacing */
        .element-container:has(.folium-map),
        .block-container,
        .main {
            padding-bottom: 0 !important;
            margin-bottom: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    #create title
    st.markdown(
        '<h1 style="font-size:28px;">Bristol Bay Wildfire Management Data Tool</h1>', 
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------
    #  define metadata - title, ee_image, colors, labels, credits
    # ---------------------------------------------------------
    recipe = {
            "Ownership": {
                "Title": "Land Ownership",
                "ee_image": ee.Image('projects/ee-azirkes1/assets/AK_proj/own_reproj').select('b1'),
                "colors": {
                    0: (255, 255, 255),   1: (141, 211, 199),   2: (255, 255, 179),   3: (190, 186, 218),
                    4: (251, 128, 114),     5: (128, 177, 211),   6: (253, 180, 98),    7: (179, 222, 105),
                    8: (252, 205, 229),     9: (139, 130, 130),   10: (204, 204, 204),  11: (204, 235, 197),
                    12: (255, 222, 111),  13: (91, 180, 199)
                                    },
                "labels": {
                    0: "No Data", 1: "BLM", 2: "Air Force", 3: "Army",
                    4: "FWS", 5: "National Park Service", 6: "State", 7: "Local Government", 8: "Private",
                    9: "FAA", 10: "Undetermined", 11: "USPS", 12: "Alaska Native Allotment", 13: "Alaska Native Lands"
                }, 
                "credits" : "Data Source: Bureau of Land Management, Alaska. n.d. "
                "BLM AK Administered Lands. ArcGIS Hub. Accessed April 01, 2025. "
                "https://gbp-blm-egis.hub.arcgis.com/ \n"
                "datasets/BLM-EGIS::blm-ak \n"
                "-administered-lands/about."
            },

            "Land Cover": {
                "Title": "LANDFIRE Land Cover",
                "ee_image": ee.Image('projects/ee-azirkes1/assets/AK_proj/landc_repro').select('b1'),
                "colors": {
                    0: (255, 255, 255), 1: (255, 255, 255),  2: (0, 0, 255), 3: (159, 161, 240), 
                    4: (253, 204, 211), 5: (255, 122, 143), 6: (1, 1, 1),       
                    7: (191, 191, 191), 8: (230, 232, 250), 9: (122, 127, 117), 
                    10: (204, 255, 153), 11: (154, 217, 108), 12: (84, 161, 53), 
                    13: (243, 213, 181), 14: (204, 147, 104),15: (191, 102, 75), 
                    16: (255, 221, 0), 17: (255, 185, 87), 18: (255, 146, 56), 
                },
                "labels": {
                    0: 'No Data', 1:  'NoData', 2: 'Water', 3: 'Snow/Ice', 4: 'Developed - Open Space', 5: 'Developed',
                    6: 'Roads', 7: 'Barren', 8: 'Quarries/Mines', 9: 'Sparse Vegetation', 10: 'Tree Cover = 10% - 24%',
                    11: 'Tree Cover = 25% - 49%', 12: 'Tree Cover = 50% - 85%', 13: 'Shrub Cover = 10% - 24%',
                    14: 'Shrub Cover = 25% - 49%', 15: 'Shrub Cover = 50% - 65%', 16: 'Herb Cover = 10% - 24%', 
                    17: 'Herb Cover = 25% - 49%', 18: 'Herb Cover = 50% - 75%'
                    },
                "credits": "Data source: LANDFIRE, 2024, Existing Vegetation Cover Layer, "
                "LANDFIRE 2.0.0, U.S. Department of the Interior, Geological Survey, "
                "and U.S. Department of Agriculture. Accessed 01 April 2025 at http://www.landfire/viewer."
                
            },

            "Wildfire Jurisdiction": {
            "Title": "Wildfire Jurisdiction",
            "ee_image": ee.Image('projects/ee-azirkes1/assets/AK_proj/jurisdic_rep6').select('b1'),
            "colors": {
                1: (165, 0, 38), #USFWS
                2: (215, 48, 39), #BLM
                3: (244, 109, 67), #BIA
                4: (253, 174, 97), #AK DNR
                5: (254, 224, 139), #NPS
                6: (255, 255, 191), #FAA
                7: (217, 239, 139), #AK DNR - Div. Parks
                8: (166, 217, 106), #ANCSA Village/AFS
                9: (102, 189, 99), #AK DOT
                10: (26, 152, 80), #ANCSA Regional/AFS
                11: (0, 104, 55), #Bourough/AKDNR
                12: (54, 144, 192), #Air Force
                13: (5, 112, 176), #Army
                14: (8, 64, 129), #uSPS
                15: (8, 29, 88), #AK Dept. Fish & Game
                16: (37, 52, 148) #City/AK DNR
            },

            "labels": {
                1: 'USFWS', 
                2: 'BLM', 
                3: 'BIA', 
                4: 'AK DNR',
                5: 'NPS',
                6: 'FAA',
                7: 'AK DNR - Div. Parks',
                8: 'ANCSA Village/AFS',
                9: 'AK DOT',
                10: 'ANCSA Regional/AFS',
                11: 'Bourough/AKDNR',
                12: 'Air Force', 
                13: 'Army', 
                14: 'USPS', 
                15: 'AK Dept. Fish & Game', 
                16: 'City/AK DNR'
            },

            "credits": "Data source: U.S. Department of the Interior, "
            "Bureau of Land Management, Alaska Fire Service."
            "Alaska Wildland Fire Jurisdictions (AKWFJ). "
            "Last modified March 3, 2023. ArcGIS Online. "
            "https://www.arcgis.com/home/item.html?\n"
            "id=1091963729c54d3386b5e60995da6fff." 
        },
        "Flammability Hazard": {
            "Title": "Flammability Hazard",
            "ee_image": ee.Image('projects/ee-azirkes1/assets/AK_proj/haz_repro_rec').select('b1'),
            "colors": {
                0: (189, 190, 190), 1: (101, 171, 20), 2: (196, 227, 29), 3: (249, 223, 26), 
                4: (255, 154, 11), 5: (252, 59, 9)
            },
            "labels": {
                0: "No data", 1: "Very Low", 2: "Low", 3: "Moderate",
                4: "High", 5: "Extreme"
            },
            "credits": "Data Source: Schmidt, Jennifer. 2025. "
                    "“Wildfire Exposure Assessment and Structure Risk.” "
                    "Alaska Natural Resource Management. Accessed April 01, 2025. "
                    "https://alaskanrm.com/wildfire-exposure/."
        }}
    
    # ---------------------------------------------------------
    #  add elements to app
    # ---------------------------------------------------------

    #explain how to select on the map 
    st.write('This tool allows a user to download relevant wildfire management data layers clipped to a region of interest. ' \
    'Simply select the data layers and data format you are interested in below. Next, draw a boundary on the map by clicking on the rectangle tool in the upper left corner of the map. ' \
    'This will be used as the clipping boundary. ' \
    'Lastly, scroll down and click the download button that appears below the map. The app may need a moment to produce the output.')

    #data layer multiselect
    with st.sidebar:
        selected_options = st.multiselect(
            "Which data layers would you like to download?",
            list(recipe.keys())
        )

    #text box for data layers
    with st.sidebar:
        st.markdown(
            """
            <div style='color: #808080; overflow: hidden;
            white-space: normal;
            word-wrap: break-word;
            margin-bottom: 15px;'>
                <u>Ownership</u> - Bureau of Land Management<br>
                <u>Land cover</u> - National Land Cover Database<br>
                <u>Wildfire Jurisdiction</u> - Bureau of Land Management<br>
                <u>Flammability Hazard</u> - University of Alaska - Anchorage<br>
            </div>
            """,
            unsafe_allow_html=True
        )

    

    #data format multiselect
    options_filetype = '.tif', '.pdf'
    with st.sidebar:
        selected_filetype = st.multiselect(
            "What format do you want the data in?",
            options_filetype
        )

    #data format text box
    with st.sidebar:
        st.markdown(
            """
            <div style='color: #808080;  overflow: hidden;
            white-space: normal;
            word-wrap: break-word;
            margin-bottom: 15px;'>
                PDFs provide an easy and simple way to view the data, whereas TIF files are ideal for both viewing and analyzing data in ArcGIS or Google Earth.
                
            </div>
            """,
            unsafe_allow_html=True
        )
    # ---------------------------------------------------------
    #  Build map and drawing tools
    # ---------------------------------------------------------

    #validate if drawing is reasonable rectangle
    def is_reasonable_rectangle(geometry, min_size=0.001, max_ratio=5, min_ratio=0.55):
        if geometry["type"] != "Polygon":
            return False
        coords = geometry["coordinates"][0]
        lons, lats = zip(*coords[:-1])
        width = max(lons) - min(lons)
        height = max(lats) - min(lats)
        area = width * height
        ratio = max(width / height, height / width) if height > 0 else float('inf')
        if area < min_size or ratio > max_ratio or ratio < min_ratio:
            return False
        return True

    #create folium map 
    m = folium.Map(location=[58.5, -157],control_scale = True, zoom_start=6, attr_control=False)
    
    #add satellite imagery 
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=True
    ).add_to(m)

    #hide attribution and Leaflet logo
    hide_attribution_css = """
    <style>
    .leaflet-control-attribution {
        display: none !important;
    }
    .leaflet-control-logo {
        display: none !important;
    }
    </style>
    """
    m.get_root().html.add_child(folium.Element(hide_attribution_css))

    #fetch PlaceNamesBBNC FeatureServer GeoJSON data for popups
    feature_server_url = "https://services7.arcgis.com/xNC9kPmqExVpPYv3/arcgis/rest/services/PlaceNamesBBNC/FeatureServer/0/query"
    params = {
        "where": "1=1",
        "outFields": "*",
        "f": "geojson"
    }
    response = requests.get(feature_server_url, params=params)
    vector_data = response.json()
    
   #add cluster to manage many popups
    cluster = MarkerCluster(name="Place Names")
    cluster.add_to(m)

    #add markers directly to the cluster 
    for feature in vector_data['features']:
        props = feature['properties']
        geom = feature['geometry']
        lon, lat = geom['coordinates']

        popup_html = f"""
        <b>Place Name:</b> {props.get('Place_Name', 'N/A')}<br>
        <b>Language:</b> {props.get('PN_Languag', 'N/A')}<br>
        <b>Type:</b> {props.get('Type', 'N/A')}
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color="cornflowerblue",
            fill=True,
            fill_opacity=0.6,
            tooltip=props.get('Place_Name', ''),
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(cluster)

    #add drawing tools to the map
    draw = Draw(
            draw_options={
                "polyline": False,
                "circle": False,
                "circlemarker": False,
                "marker": False,
                "rectangle": True,
                "polygon": False,
            },
            edit_options={"edit": True, "remove": True},
        )
    draw.add_to(m)
    
    #get bbnc boundary from Google Earth Engine 
    bbnc = ee.FeatureCollection('projects/ee-azirkes1/assets/AK_proj/bbnc_boundary')

    #turn bbnc boundary into geojson geometry and add to map
    geojson_dict = bbnc.geometry().getInfo()
    folium.GeoJson(
        geojson_dict,
        name="BBNC Boundary",
        style_function=lambda feature: {
            'fillColor': "#ffffff00",
            'color': "#080808",
            'weight': 5,
            'fillOpacity': 0,
        }
    ).add_to(m)

    #add layer toggle control
    folium.LayerControl(collapsed=False).add_to(m)
    
    #render map and capture drawing events
    map_result = st_folium(m, height=500, width=700, returned_objects=["all_drawings"])

    # ---------------------------------------------------------
    #  main function, returns pdf, tif, metadata
    # ---------------------------------------------------------

    def img(recipe: dict, roi: ee.Geometry, layer_name):
        final_output = None

        # ---------------------------------------------------------
        #  set up PDF helper functions 
        # ---------------------------------------------------------
        
        #function to get metadata for layer and write it to text 
        def generate_text_metadata_file(recipe: dict, layer_name: str) -> bytes: 

            #attempts to find layer_name in recipe keys
            matched_key = next((k for k in recipe if k.strip().lower() == layer_name.strip().lower()), None) 
            if matched_key is None:
                return b""  #return empty bytes 

            #get metadata for the layer
            layer_recipe = recipe.get(matched_key, {}) 
            classes = layer_recipe.get("labels", {})
            credits = layer_recipe.get("credits", "")
            symbology = layer_recipe.get("colors", {})

            #writes metadata to text and returns it 
            metadata_lines = [
                f"Layer: {matched_key}",
                f"Credits: {credits}",
                "Classes:",
                *[f"  - {k}: {v}" for k, v in classes.items()],
                "Symbology:",
                *[f"  - {k}: RGB{v}" for k, v in symbology.items()]
            ]

            text = "\n".join(metadata_lines)
            return text.encode("utf-8")
        
        #function to calculate bounding box from coordinates
        def _min_max_coords(coords): 
                    xs, ys = zip(*coords)
                    return min(xs), min(ys), max(xs), max(ys)
        
        #function to remove duplicate legend entries and return dict of colors and labels that are in present_vals
        def de_duplicate_entries(colors_dict, labels_dict, present_vals): 
            seen = set()
            deduped_colors = {}
            deduped_labels = {}
            for val in present_vals:
                if val in colors_dict:
                    rgb = colors_dict[val]
                    label = labels_dict.get(val, str(val))
                    key = (rgb, label)
                    if key not in seen:
                        seen.add(key)
                        deduped_colors[val] = rgb
                        deduped_labels[val] = label
            return deduped_colors, deduped_labels
        
        #function to calculate aspect ratio
        def get_aspect_ratio(geom): 
            coords = geom.bounds().getInfo()["coordinates"][0]
            xs, ys = zip(*coords)
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            return width / height if height > 0 else float('inf')

        #function to build legend 
        def build_legend_image(colors, labels, present, map_h): 

            #use De_duplicate to get colors and labels
            colors, labels = de_duplicate_entries(colors, labels, present)
            shown_keys = sorted(colors.keys())

            #set font size based on map height
            scale = map_h / 1000
            font_size = max(int(12 * scale), 12)

            patches = []
            for k in shown_keys:

                #retrives rgb tuple and normalizes it 
                rgb = tuple(colors[k])
                normalized_color = [v / 255 for v in rgb]

                #if it's white, create patch with a black edge
                if rgb == (255, 255, 255):
                    patch = mpatches.Patch(
                        facecolor=normalized_color,
                        edgecolor='black',
                        linewidth=0.5,
                        label=labels.get(k, str(k))
                    )
                else: 
                    #create patch with defaults
                    patch = mpatches.Patch(
                        color=normalized_color,
                        label=labels.get(k, str(k))
                    )
                #add patches to list 
                patches.append(patch)

            #create a blank figure and adds legend and patches
            fig = plt.figure()
            legend = fig.legend(
                handles=patches,
                loc="center",
                frameon=False,
                fontsize=font_size,
                borderpad=0.3,
                handlelength=1.5,
                handletextpad=0.6
            )

            #adds legend to canvas
            canvas = FigureCanvas(fig)
            fig.set_size_inches(6, len(patches) * 0.4 + 0.4)  #scales the height depending on number of patches 
            canvas.draw()

            #crop legend to content 
            bbox = legend.get_window_extent() #gets bounding box
            bbox = bbox.expanded(1.05, 1.05)  #slight padding
            image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').reshape(canvas.get_width_height()[::-1] + (4,)) #extractsRGBA buffer and formats it in numpy array 
            legend_img = Image.fromarray(image) #converts to PIL image
            legend_img = legend_img.crop((int(bbox.x0), int(bbox.y0), int(bbox.x1), int(bbox.y1))) #crop legend

            plt.close(fig)
            return legend_img
        
        #function to build scalebar
        def add_scalebar_from_ax_extent(ax, location='lower left'):

            #get ax extent and use it to set the length of scalebar
            xmin, xmax = ax.get_xlim()
            map_width_m = abs(xmax - xmin)
            scalebar_length = map_width_m / 3  

            #set scalebar properties
            scalebar = ScaleBar(
                dx=1,
                units='m',
                length_fraction=scalebar_length / map_width_m,
                location=location,
                box_alpha=0.7,
                color='black',
                font_properties="Arial"
            )
            ax.add_artist(scalebar)
        
        #function to build locator map 
        def create_locator_map(clipped_geom, width=450, height=450, dpi=150): 
        
            # Get full and clipped bounds from GEE using owner_raster and clipped_geom
            full_extent = ee.Image('projects/ee-azirkes1/assets/AK_proj/own_reproj').geometry()
            full_bounds = full_extent.bounds().getInfo()['coordinates'][0]
            clip_bounds = clipped_geom.bounds().getInfo()['coordinates'][0]

            #convert to shapely polygons
            full_poly = box(*_min_max_coords(full_bounds))
            clip_poly = box(*_min_max_coords(clip_bounds))

            #convert to GeoDataFrames in Web Mercator
            gdf_full = gpd.GeoDataFrame(geometry=[full_poly], crs="EPSG:4326").to_crs(epsg=3857)
            gdf_clip = gpd.GeoDataFrame(geometry=[clip_poly], crs="EPSG:4326").to_crs(epsg=3857)

            #get center of clipped geometry for dot
            center = gdf_clip.geometry.centroid.iloc[0]

            #set up figure
            figsize = (width / dpi, height / dpi)
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            #plot full boundary
            gdf_full.boundary.plot(ax=ax, color='black')

            # Set axis limits exactly to polygon bounds (in EPSG:3857)
            bounds = gdf_full.total_bounds  # [minx, miny, maxx, maxy]
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])

            # Remove any padding/margins
            ax.margins(0)

            # Turn off axes and ticks as before
            ax.set_axis_off()

            #add red dot at center of ROI
            ax.plot(center.x, center.y, 'o', color='red', markersize=4)

            #add basemap
            cx.add_basemap(ax, source=cx.providers.CartoDB.Voyager, attribution=False)

            #final formatting
            ax.set_axis_off()
            ax.set_title("Locator Map", fontsize=10)

            #export to image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)

            return Image.open(buf)

        #function to add credits below main map 
        def append_credits_below( 
            image,
            text,
            font_path=None,
            font_size=20,
            side_padding=40,
            top_padding=0,
            bottom_padding=0,
            line_spacing=4
        ): 
            if not text or not text.strip(): #if no credits, return image
                return image

            #load font 
            try:
                font = ImageFont.truetype(font_path or "arial.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()
            temp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))

            #text wrapping 
            max_width = image.width - 2 * side_padding
            words = text.split()
            lines = []
            current_line = ""
            for word in words:
                test_line = f"{current_line} {word}".strip()
                if temp_draw.textlength(test_line, font=font) <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

            if not lines:
                return image  

            #measures line and text block height
            line_height = font.getbbox("Ay")[3] + line_spacing
            text_block_height = line_height * len(lines)

            #create blank image with correct size
            new_height = image.height + top_padding + text_block_height + bottom_padding
            new_img = Image.new("RGB", (image.width, new_height), "white")
            new_img.paste(image, (0, 0))

            #add credits text 
            draw = ImageDraw.Draw(new_img)
            y = image.height + top_padding
            for line in lines:
                x = side_padding
                draw.text((x, y), line, font=font, fill="black")
                y += line_height
            return new_img
        
        #function to horizontally concatenate image into one row 
        def concat_horizontal(images, align='top', padding=10, bg_color=(255, 255, 255, 255)): 

            #calculates dimensions across all images
            heights = [img.height for img in images]
            total_width = sum(img.width for img in images) + padding * (len(images) - 1)
            max_height = max(heights)

            #creates new blank image
            new_img = Image.new("RGBA", (total_width, max_height), bg_color)
            
            #loops through images and pastes them 
            x = 0
            for img in images:
                if align == 'center':
                    y = (max_height - img.height) // 2
                elif align == 'bottom':
                    y = max_height - img.height
                else:  # 'top' or default
                    y = 0
                new_img.paste(img, (x, y))
                x += img.width + padding

            return new_img

        #function to vertically concatenate images into one column 
        def concat_vertical(images, align='left', padding=10): 

            #calculates dimensions across all images
            widths = [img.width for img in images]
            max_width = max(widths)
            total_height = sum(img.height for img in images) + padding * (len(images)-1)

            #creates new blank image
            new_img = Image.new("RGBA", (max_width, total_height), (255, 255, 255, 255))

            #loops through images and pastes them 
            y = 0
            for img in images:
                if align == 'center':
                    x = (max_width - img.width) // 2
                elif align == 'right':
                    x = max_width - img.width
                else:
                    x = 0
                new_img.paste(img, (x, y))
                y += img.height + padding
            return new_img
        
        #function to trim whitespace from image
        def trim_whitespace(im, bg_color=(255, 255, 255)): 
            bg = Image.new(im.mode, im.size, bg_color)
            diff = ImageChops.difference(im, bg)
            bbox = diff.getbbox()
            return im.crop(bbox) if bbox else im
        
        # ---------------------------------------------------------
        #  set up TIF helper functions 
        # ---------------------------------------------------------
        
        #function to write metadata dext file 
        def generate_text_metadata_file(recipe: dict, layer_name: str) -> bytes:
            matched_key = next((k for k in recipe if k.strip().lower() == layer_name.strip().lower()), None)
            if matched_key is None:
                return b""  # Return empty bytes

            layer_recipe = recipe.get(matched_key, {})
            classes = layer_recipe.get("labels", {})
            credits = layer_recipe.get("credits", "")
            symbology = layer_recipe.get("colors", {})

            metadata_lines = [
                "=" * 20,
                f"Layer: {matched_key}",
                "=" * 20,
                f"Credits: {credits or 'N/A'}",
                "",
                "Classes:",
                *[f"  - {k}: {v}" for k, v in classes.items()],
                "",
                "Symbology:",
                *[f"  - {k}: RGB{v}" for k, v in symbology.items()],
                ""  # final line break
            ]

            text = "\n".join(metadata_lines)
            return text.encode("utf-8")
        
        #function to extract tif from GEE zip file 
        def extract_tif_from_zip(zip_bytes):
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                for name in zf.namelist():
                    if name.endswith(".tif") or name.endswith(".tiff"):
                        return zf.read(name)
                raise ValueError("No .tif file found in the ZIP archive.")
            
       

        # ---------------------------------------------------------
        #  back to main img function - extracting TIF data
        # ---------------------------------------------------------
        
        #extract just the selected layer's recipe
        layer_recipe = recipe[layer_name] 
       
        #create metadata text file 
        txt_bytes = generate_text_metadata_file(recipe, layer_name) 

        #get image to google earth engine and cast to int
        img_ee = layer_recipe["ee_image"].clip(roi).unmask(0).toInt()

        #generate download URL with nearest resampling
        tiff_url = img_ee.getDownloadURL({
            'scale': 30,
            'crs': 'EPSG:3338',
            'region': roi.getInfo()['coordinates'],
            'filePerBand': False,
            'formatOptions': {
                'resampling': 'nearest'
            }
        })
            
        #sends HTTP GET request and returns ZIP file
        response = requests.get(tiff_url)

        #reads the ZIP file and extract the tif file
        zip_bytes = response.content
        original_tif = extract_tif_from_zip(zip_bytes)

        #pull colors and labels from layer recipe 
        cmap = layer_recipe["colors"]
        labels = layer_recipe["labels"]

        #get aspect ratio 
        aspect = get_aspect_ratio(roi)
        
        #get layout type based on aspect ratio 
        if aspect > 3:
            layout = "horizontal"  #wide → legend/locator below
        elif aspect < 1.5:
            layout = "vertical"  #tall → legend/locator right
        else:
            layout = "square"  #square → default

        # ---------------------------------------------------------
        #  reproject tif 
        # ---------------------------------------------------------
        
        #wraps tif_bytes into in-memory file
        with MemoryFile(io.BytesIO(original_tif)) as mem: 
            #opens tif file and reads it 
            with mem.open() as src:
                band_data = src.read(1)
                width = src.width
                height = src.height
                dtype = src.dtypes[0]
                count = src.count
                profile = src.profile.copy()

                #gets bounding box from geometry 
                lonlat_coords = geometry['coordinates'][0]
                lons, lats = zip(*lonlat_coords)
                xmin, xmax = min(lons), max(lons)
                ymin, ymax = min(lats), max(lats)

                #reprojects bounding box from EPSG:4326 to to EPSG:3338 and 
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:3338", always_xy=True)
                x0, y0 = transformer.transform(xmin, ymin)
                x1, y1 = transformer.transform(xmax, ymax)

                #define raster transform and crs
                dst_crs = CRS.from_epsg(3338)
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )

            #update the profile with CRS and transform
                profile.update({
                    'driver': 'GTiff',
                    'height': height,
                    'width': width,
                    'count': count,
                    'dtype': dtype,
                    'crs': dst_crs,
                    'transform': transform,
                })

                #create empty base file
                destination = np.empty((height, width), dtype=src.dtypes[0])

                #reproject
                reproject(
                    source=src.read(1),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )

                # Write the new raster
                fixed_memfile = MemoryFile()
                with fixed_memfile.open(**profile) as dst:
                    dst.write(band_data, 1)

                tif_bytes = fixed_memfile.read()

            # ---------------------------------------------------------
            #  create main pdf map 
            # ---------------------------------------------------------
            with MemoryFile(io.BytesIO(tif_bytes)) as mem:
                with mem.open() as src:
                
                    #read the band and create a mask for nodata values
                    band = src.read(1)
                    nodata = src.nodata or 0
                    masked_band = np.ma.masked_equal(band, nodata) #create mask that excludes nodata

                    #convert raster to RGB image 
                    rgb = np.ones((masked_band.shape[0], masked_band.shape[1], 3), dtype=np.uint8) * 255
                    for k, color in cmap.items():
                        rgb[masked_band == k] = color

                    #set projection and extent 
                    proj = ccrs.epsg(3338)
                    extent = [x0, x1, y0, y1]

                    #set figure size based on layout 
                    if layout == "vertical":
                        fig_size = (8, 11)
                    elif layout == "horizontal":
                        fig_size = (11, 8)
                    else:
                        fig_size = (10, 10)

                    #plot the image
                    fig, ax = plt.subplots(figsize=fig_size, subplot_kw={'projection': proj}) #create figure
                    ax.imshow(rgb, origin='upper', extent=extent) #render image
                    ax.set_extent(extent, crs=proj) #set axis extent 
                    ax.set_title(f"{layer_name} Map", fontsize=18) #create title 
                    ax.set_aspect('equal')

                    #add lat/lon gridlines in degrees
                    gl = ax.gridlines(
                        crs=ccrs.PlateCarree(),
                        draw_labels=True,
                        linewidth=0.8,
                        color='gray',
                        alpha=0.7,
                        linestyle='--'
                    )

                    #hide labels on top right
                    gl.top_labels = False 
                    gl.right_labels = False 

                    #set font size for x and y axis labels
                    gl.xlabel_style = {'size': 10}
                    gl.ylabel_style = {'size': 10}

                    #estimate image width in pixels
                    width = src.width
                    height = src.height

                    add_scalebar_from_ax_extent(ax)
                                        
                    #saves figure as PNG in memory and loads it as a PIL image
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    buf.seek(0)
                    map_img = Image.open(buf)
                            
            # ---------------------------------------------------------
            #  build legend and locator map
            # ---------------------------------------------------------
            #get present classes
            present = set(np.unique(band)) 

            #create legend and locator
            legend_img = build_legend_image(cmap, labels, present, map_img.height)
            locator_img = create_locator_map(roi)

            #trim whitespace
            legend_img = trim_whitespace(legend_img)
            locator_img = trim_whitespace(locator_img)

            #pad smaller width to match wider one 
            legend_w, locator_w = legend_img.width, locator_img.width #get widths
            target_width = max(legend_w, locator_w) #get max width
            if legend_w < target_width:
                pad = (target_width - legend_w) // 2
                legend_img = ImageOps.expand(legend_img, border=(pad, 0, target_width - legend_w - pad, 0), fill="white")
            if locator_w < target_width:
                pad = (target_width - locator_w) // 2
                locator_img = ImageOps.expand(locator_img, border=(pad, 0, target_width - locator_w - pad, 0), fill="white")

            #stack legend and locator
            stacked = concat_vertical([legend_img, locator_img], align="center", padding=10)
            
            #get credits
            credit_text = recipe[layer_name].get("credits", "")
            font_size = 15

            # ---------------------------------------------------------
            #  layout logic
            # ---------------------------------------------------------
            if layout == "horizontal":

                #pad to center vertically relative to map height
                if stacked.height < map_img.height:
                    pad_top = (map_img.height - stacked.height) // 2
                    pad_bottom = map_img.height - stacked.height - pad_top
                    stacked = ImageOps.expand(stacked, border=(0, pad_top, 0, pad_bottom), fill="white")

                #combine map and stacked legend/locator
                combined = concat_horizontal([map_img, stacked], align="top")
                
                #add credits below the map
                final_output = append_credits_below(
                    combined,
                    credit_text,
                    font_size=font_size,
                    bottom_padding=30
                )

            elif layout in ["vertical", "square"]:

                #add padding above stacked
                extra_padding_top = 60  # pixels of padding you want
                stacked = ImageOps.expand(stacked, border=(0, extra_padding_top, 0, 0), fill="white")

                #add credits below the stacked legend/locator
                legend_with_credits = append_credits_below(
                    stacked,
                    credit_text,
                    font_size=font_size,
                    bottom_padding=5
                )
                
                #combine map and legend/locator/credits
                final_output = concat_horizontal([map_img, legend_with_credits], align="top")
                
            #export to pdf
            pdf_bytes = io.BytesIO()
            if final_output:
                final_output.save(pdf_bytes, "PDF", dpi=(300, 300))
                pdf_bytes.seek(0)
            else:
                #fallback blank PDF or raise error
                img = Image.new("RGB", (100, 100), color="white")
                img.save(pdf_bytes, "PDF", dpi=(300, 300))
                pdf_bytes.seek(0)

            #return all outputs
            return pdf_bytes, io.BytesIO(tif_bytes), txt_bytes
        
    # ---------------------------------------------------------
    # access draw data and combine clipped data into zip folder
    # ---------------------------------------------------------
    
    if (
    selected_filetype
    and any(ft.strip() for ft in selected_filetype)
    and selected_options
    and any(opt.strip() for opt in selected_options)
    and map_result.get("all_drawings")
    ):
        
        all_drawings = map_result["all_drawings"]
        last_drawing = all_drawings[-1]
        geometry = last_drawing.get("geometry")

        if geometry is None:
            st.error("Please draw a rectangle boundary on the map.")
            st.stop()

        polygon = ee.Geometry.Polygon(geometry["coordinates"])

        # Check if within BBNC
        bbnc_geom = bbnc.geometry()
        if not bbnc_geom.contains(polygon).getInfo():
            st.error("Your drawing is outside the BBNC boundary. Please draw within the designated area.")
            st.stop()

        # --- ZIP + download logic ---
        zip_buffer = io.BytesIO()
        all_metadata = []

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
            for layer_name in selected_options:
                layer_name = layer_name.strip()
                if layer_name not in recipe:
                    continue
                pdf_bytes, tif_bytes, txt_bytes = img(recipe, polygon, layer_name)
                if '.pdf' in selected_filetype:
                    z.writestr(f"{layer_name.lower().replace(' ', '_')}.pdf", pdf_bytes.getvalue())
                if '.tif' in selected_filetype:
                    z.writestr(f"{layer_name.lower().replace(' ', '_')}.tif", tif_bytes.getvalue())
                if txt_bytes:
                    decoded = txt_bytes.decode("utf-8")
                    all_metadata.append(decoded)

            if all_metadata:
                joined_metadata = "\n\n".join(all_metadata)
                z.writestr("metadata.txt", joined_metadata.encode("utf-8"))

        zip_buffer.seek(0)

        st.download_button(
            label="Download Files",
            data=zip_buffer,
            file_name="BristolBay_Wildfire.zip",
            mime="application/zip"
        )
    else:
        # Handle specific errors here for clarity
        if not selected_filetype or all(ft.strip() == "" for ft in selected_filetype):
            st.error("Please select a file format.")
        if not selected_options or all(opt.strip() == "" for opt in selected_options):
            st.error("Please select a data layer.")
        if not map_result.get("all_drawings"):
            st.error("Please draw a rectangle on the map.")