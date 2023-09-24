# Convert the coordinates of Sentinel-2 images to shapefiles.

lat = []
lon = []
coordin = "105.67262889968784 20.814616746051726,105.94926271143514 20.814616746051726,105.94926271143514 21.18673082127006,105.67262889968784 21.18673082127006,105.67262889968784 20.814616746051726"
list_str1 = coordin.split(',')  # '121.04216560657493 30.745609076526918'
for i in range(len(list_str1)):
    list_str2 = list_str1[i].split(' ')
    lat.append(list_str2[1])  # lat：'30.745609076526918'
    lon.append(list_str2[0])  # lon：'121.04216560657493'

print("var geometry = ee.Geometry.Polygon([[")
print("[" + lon[0] + ", " + lat[2] + "], ")
print("[" + lon[0] + ", " + lat[0] + "], ")
print("[" + lon[1] + ", " + lat[0] + "], ")
print("[" + lon[1] + ", " + lat[2] + "]]]);")

