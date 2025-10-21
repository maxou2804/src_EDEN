import geopandas as gpd
import requests
import zipfile
import io
import matplotlib.pyplot as plt

# URL correcte du shapefile Natural Earth
url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"

# Télécharger le fichier zip en mémoire
r = requests.get(url)
r.raise_for_status()  # vérifier que le téléchargement a fonctionné

# Ouvrir le zip depuis la mémoire
z = zipfile.ZipFile(io.BytesIO(r.content))

# Extraire le contenu dans un dossier local
z.extractall("natural_earth_data")  # crée le dossier natural_earth_data

# Lire le shapefile avec GeoPandas
world = gpd.read_file("natural_earth_data/ne_110m_admin_0_countries.shp")

print(world.head())
world.plot()
