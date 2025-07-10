import folium
import geocoder
from utillc import *
import pandas as pd
import os, sys, glob
import tqdm, re
import pickle



files = glob.glob("/mnt/NUC/download/List*Ventes*.pdf")
EKOX(files)

def extract(file) :
	os.system("pdftotext -layout '%s' out.txt" % file)
	with open("out.txt") as f:
		lines = f.readlines()
		def x(l) :
			try :
				spaces, word, number = " +", "([\w/-]+)", "([\d]+)"
				match = re.search(word + spaces + word + spaces + word + spaces + word, l)
				ville = match.group(4)
				_, e = match.span()
				match = re.search('(' + number + "/" + number + '/' + number + ")", l)
				date = match.group(1)
				e1, e3 = match.span()
				add = l[e:e1].strip()
				l3 = l[e3:]
				match = re.search(word + spaces + word + spaces, l3)
				_, e4 = match.span()
				l4 = l3[e4:]
				match = re.search(word + spaces + word+ spaces + word+ spaces + word, l4)
				prix = match.group(1)+match.group(2)
				surface = match.group(4)
				#EKON(ville, date, prix, surface, add)

				return add +  "; " + ville + "; FRANCE", date, prix, surface
			except :
				return None
		return [ e for e in map(x, lines) if e is not None]

ll = [e for l in list(map(extract, files)) for e in l]
#EKOX(ll)
EKOX(len(ll))
add=ll[0]
a, date, prix, surface = add
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


lfn = "latlong.pckl"
EKOX(os.path.exists(lfn))

if not os.path.exists(lfn) :
	latlong = [ geocoder.arcgis(e[0]).latlng for e in tqdm.tqdm(ll)]
	with open(lfn, 'wb') as handle:
		pickle.dump(latlong, handle, protocol=pickle.HIGHEST_PROTOCOL)
else :
	with open(lfn, 'rb') as handle:
		latlong = pickle.load(handle)	


dd = {}

for add, elatlong in tqdm.tqdm(zip(ll, latlong)) :
	a, date, prix, surface = add
	if a not in dd :
		dd[a] = []
	#EKOX(elatlong)
	dd[a].append((date, prix, surface, elatlong, a))
	


for a,vs in dd.items() :
	def tt(x) :
		date, prix, surface, latlong, add = x
		return date + ", " + str(prix) + "€, " + str(surface) + "m2"
	add = vs[0][4]

	EKOX(vs[0][0])
	EKOX("-".join(vs[0][0].split("/")[::-1]))
	vs = sorted(vs, key = lambda x : "-".join(x[0].split("/")[::-1]))
	txt = add + '<br>' + '<br>'.join(map(tt, vs))
	elatlong = vs[0][3]
	#EKOX(elatlong)
	lat, lng = elatlong
	folium.CircleMarker(
        [lat, lng],
        radius=5,
		tooltip = txt,
        popup=txt,
        fill=True,
        color='Blue',
        fill_color='Yellow',
        fill_opacity=0.6
        ).add_to(m)


m.save("footprint.html")
