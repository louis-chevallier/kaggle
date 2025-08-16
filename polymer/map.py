import folium
import geocoder
from utillc import *
import pandas as pd
import os, sys, glob
import tqdm, re
import pickle
from collections import defaultdict
import hashlib
import seaborn as sns
import matplotlib.pyplot as plt
import datetime


def myhash(s) :
		"""
		because default hash produces diff values on successive exec ( random seed) 
		"""
		hash_obj = hashlib.sha256(s.encode('utf-8'))
		hex_hash = str(hash_obj.hexdigest())
		return hex_hash

files = glob.glob("/mnt/NUC/download/List*Ventes*.pdf")
files = glob.glob("/tmp/erquy/List*Ventes*.pdf")
EKOX(files)

def extract(file) :
	os.system("pdftotext -layout '%s' out.txt" % file)
	EKOX(file)
	with open("out.txt") as f:
		lines = f.readlines()
		def x(l) :
			try :
				spaces, word, numbers = " +", "([\w/-]+)", "([\d]+)"
				match = re.search(word + spaces + word + spaces + word + spaces + word, l)
				ville = match.group(4)
				_, e = match.span()
				match = re.search('(' + numbers + "/" + numbers + '/' + numbers + ")", l)
				date_transaction = match.group(1)
				date_transaction = datetime.datetime.strptime(date_transaction, "%d/%m/%Y").date()
				
				e1, e3 = match.span()
				match = re.search('(' + numbers + "/" + numbers + '/' + numbers + ")" + spaces + numbers, l)
				annee_construction = int(match.group(5))
				add = l[e:e1].strip()
				l3 = l[e3:]
				match = re.search(word + spaces + word + spaces, l3)
				_, e4 = match.span()
				l4 = l3[e4:]
				match = re.search(word + spaces + word+ spaces + word+ spaces + word, l4)
				prix = int(match.group(1) + match.group(2))
				surface = int(match.group(4))
				match = re.search(word + spaces + word+ spaces + word+ spaces + word, l4)				
				terrain = int(match.group(3))
				#EKON(ville, date_transaction, annee_construction, prix, surface, terrain, add)
				

				return add +  "; " + ville + "; FRANCE", date_transaction, prix, surface, annee_construction, terrain
			except Exception as e:
				#EKOX(l)	
				#EKOX(e)
				return None
		return [ e for e in map(x, lines) if e is not None]

ll = [e for l in list(map(extract, files)) for e in l]
#EKOX(ll)
EKOX(len(ll))
add=ll[0]
a, date, prix, surface, annee, terrain = add
EKOX(a)

g = geocoder.arcgis(a).latlng
EKOX(g)
m = folium.Map(location=g,
			   zoom_start=14,
			   tooltip = 'This tooltip will appear on hover',
			   tiles="cartodb positron")

tile = folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
       ).add_to(m)

#ll = [ (a, date, prix, surface, annee, terrain) for a, date, prix, surface, annee, terrain in ll if "GARENNE" in a]

EKOX(myhash(str(ll)))

lfn = "latlong_%s.pckl" % myhash(str(ll))
EKOX(lfn)
EKOX(os.path.exists(lfn))

default_color_1 = 'darkblue'
default_color_2 = 'darkgreen'
default_color_3 = 'darkred'

if not os.path.exists(lfn) :
	latlong = [ geocoder.arcgis(e[0]).latlng for e in tqdm.tqdm(ll)]
	with open(lfn, 'wb') as handle:
		pickle.dump(latlong, handle, protocol=pickle.HIGHEST_PROTOCOL)
else :
	with open(lfn, 'rb') as handle:
		latlong = pickle.load(handle)	


dd = {}

dd = defaultdict(list)

def sel(add, elatlong) :
	a, date, prix, surface, annee, terrain = add
	return a, (date, prix, surface, elatlong, a)
		
[ sel(add, elatlong) for add, elatlong in tqdm.tqdm(zip(ll, latlong)) ]





for add, elatlong in tqdm.tqdm(zip(ll, latlong)) :
	a, date, prix, surface, annee, terrain = add

	#if a not in dd :		dd[a] = []
	
	EKON(add, elatlong)
	dd[a].append((date, prix, surface, elatlong, a))
	
for a,vs in dd.items() :
	def tt(x) :
		date, prix, surface, latlong, add = x
		return str(date) + ", " + str(prix) + "€, " + str(surface) + "m2"
	add = vs[0][4]

	#EKOX(vs[0][0])
	#EKOX("-".join(vs[0][0].split("/")[::-1]))
	#vs = sorted(vs, key = lambda x : "-".join(x[0].split("/")[::-1]))
	vs = sorted(vs, key = lambda x : x[0].year)
	txt = add + '<br>' + '<br>'.join(map(tt, vs))

	date, prix, surface, elatlong, add = vs[0]	
	colors = ["Yellow", "Orange", "Red"]
	col = colors[ min( prix//150000, 2)]
	#EKOX(elatlong)
	lat, lng = elatlong
	folium.CircleMarker(
        [lat, lng],
        radius=5,
		tooltip = txt,
        popup=txt,
        fill=True,
        color=col,
        fill_color='Blue',
        fill_opacity=0.6
        ).add_to(m)


m.save("/mnt/NUC/www/erquy.html")
df = pd.DataFrame(ll, columns=["a", "date", "prix", "surface", "annee", "terrain"])
EKOX(df)
EKOX(df[["prix", "surface"]])
sns.pairplot(data=df[["prix", "surface", "terrain", "annee", "date"]],
             diag_kws = { 'color' : default_color_3},
             plot_kws = { 'color' : default_color_3,
                          'alpha' : 0.5,
                          's' : 15})
plt.show()

#sns.regplot(x=pd.to_datetime(df["date"]).dt.strftime('%Y:%m:%d'), y=df["prix"], ci=None, color="r")
plt.scatter(df["date"], df["prix"])
plt.show()

