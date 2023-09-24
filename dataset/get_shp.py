# Convert the coordinates of Sentinel-2 images to shapefiles.

import fiona


def switch_shp():
    schema = {
        'geometry': 'Polygon',
        # 'properties': [('Name', 'str')]
    }
    # open a fiona object
    polyShp = fiona.open('E:/data_tmp/WaterVec/arcmap/shp/VN.shp', mode='w', driver='ESRI Shapefile',
                         schema=schema, crs="EPSG:4326")
    # get list of points
    xyList = [(105.67262889968784, 21.18673082127006),
              (105.67262889968784, 20.814616746051726),
              (105.94926271143514, 20.814616746051726),
              (105.94926271143514, 21.18673082127006)]
    # save record and close shapefile
    rowDict = {
        'geometry': {'type': 'Polygon',
                     'coordinates': [xyList]},  # Here the xyList is in brackets
        # 'properties': {'Name': rowName},
    }
    polyShp.write(rowDict)
    # close fiona object
    polyShp.close()


if __name__ == "__main__":
    switch_shp()
