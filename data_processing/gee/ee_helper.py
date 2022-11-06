
from httplib2 import Credentials
import numpy as np
import json
import rasterio.features
import rasterio
import rasterio.warp
import ee
import requests
from retry import retry
import shutil
import multiprocessing
import math
import os


def getPoints(path, tif_band, tif_image, min_prob):
    p8list = []
    band1 = tif_image.read(1)
    height = band1.shape[0]
    width = band1.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(tif_image.transform, rows, cols)
    lons = np.array(xs)
    lats = np.array(ys)
    for row in range(0, height):
        for col in range(0, width):
            p = tif_band[row][col]
            if p >= min_prob and p <= 1.0:
                p8list.append([lons[row][col], lats[row][col]])
    print(f"tif size: {width} x {height}")
    print(len(p8list))
    jsono = {
        'p': min_prob,
        'count': len(p8list),
        'lat': [],
        'lon': [],
    }
    for i in range(len(p8list)):
        jsono['lon'].append(p8list[i][0])
        jsono['lat'].append(p8list[i][1])

    with open(path, 'w') as outfile:
        json.dump(jsono, outfile, indent=4)

    return p8list


def print_parm(path, index, elv, points, dimensions=[128, 128], meters=13000):
    print(path)
    print(index)
    print(points)

# see  https://developers.google.com/earth-engine/guides/service_account
eejson = "../../ee.json" # '.private-key.json'
s_account = 'my-service-account@...gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(s_account, eejson)
ee.Initialize(credentials=credentials,
              opt_url='https://earthengine-highvolume.googleapis.com')
RGB = ['B4', 'B3', 'B2']
PERCENTILE_SCALE = 50
NEWRGB = ['B4_median', 'B3_median', 'B2_median']
TRUE_RGB = ['TCI_R', 'TCI_G', 'TCI_B']


class EEH:
    def __init__(self, directory = './output', size = 512):

        self.dimensions = [size, size]
        self.path = directory
        self.dem_path = directory +'/dem'
        self.alos_path = directory +'/alos'
        self.sat_path = directory + '/sat'
        self.meters = 24000
        self.min = 0
        self.max = 3000
        self.sat_min = 0
        self.sat_max = 300
        self.crs = 'EPSG:4326'
        self.elv = ee.Image('CGIAR/SRTM90_V4').unmask(1000)

        self.init_directory()
        
    def init_directory(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if not os.path.exists(self.dem_path):
            os.makedirs(self.dem_path)
        if not os.path.exists(self.sat_path):
            os.makedirs(self.sat_path)
        if not os.path.exists(self.alos_path):
            os.makedirs(self.alos_path)


    @retry(tries=10, delay=1, backoff=2)
    def getDEM(self, index, points):
        u_lon = points[0]
        u_lat = points[1]
        u_poi = ee.Geometry.Point(u_lon, u_lat)
        lyon = u_poi.buffer(self.meters)  # meters
        url = self.elv.getThumbUrl({'min': self.min,
                                    'max': self.max,
                                    'region': lyon,
                                    'dimensions': self.dimensions,
                                    'bestEffort': True,
                                    'crs': self.crs,
                                    'format': 'png'
                                    })

        r = requests.get(url, stream=True)
        if r.status_code != 200:
            print("Failed to get image...")
            raise r.raise_for_status()
        filename = self.dem_path+ '/%05d.png' % index
        with open(filename, 'wb') as out_file:
            shutil.copyfileobj(r.raw, out_file)
        print(f"Done: {filename}")

    def workDEM(self, indexes, points):
        pool = multiprocessing.Pool()
        pool.starmap(self.getDEM, zip(indexes, points))
        pool.close()
        pool.join()
        
    @retry(tries=10, delay=1, backoff=2)
    def getALOSDEM(self, index, points):
        u_lon = points[0]
        u_lat = points[1]
        u_poi = ee.Geometry.Point(u_lon, u_lat)
        lyon = u_poi.buffer(self.meters)  # meters
        alos = ee.ImageCollection('JAXA/ALOS/AW3D30/V3_2').select('DSM')#.filterBounds(lyon)

        proj = alos.first().select(0).projection()

        alos = alos.mosaic().setDefaultProjection(proj)
        url = alos.getThumbUrl({'min': self.min,
                                    'max': self.max,
                                    'region': lyon,
                                    'dimensions': self.dimensions,
                                    'format': 'png',
                                    'crs': self.crs,
                                    'bestEffort': True,
                                    })

        r = requests.get(url, stream=True)
        
        if r.status_code != 200:
            print("Failed to get image...")
            raise r.raise_for_status()
        filename = self.alos_path+ '/%05d.png' % index
        with open(filename, 'wb') as out_file:
            shutil.copyfileobj(r.raw, out_file)
        print(f"Done: {filename}")

    def workALOS(self, indexes, points):
        pool = multiprocessing.Pool()
        pool.starmap(self.getALOSDEM, zip(indexes, points))
        pool.close()
        pool.join()

    @retry(tries=10, delay=1, backoff=2)
    def getSAT(self, index, points):
        u_lon = points[0]
        u_lat = points[1]
        u_poi = ee.Geometry.Point(u_lon, u_lat)
        lyon = u_poi.buffer(self.meters)  # meters
        # camel
        # image= (ee.ImageCollection("COPERNICUS/S2_SR")
        #             .filterBounds(lyon.bounds())  
        #             # .filterDate('2019-11-15', '2019-12-15')

        #              .filter(ee.Filter.calendarRange(2019, 2020,  'year'))
        #             .filter(ee.Filter.calendarRange(11, 2, 'month'))
        #              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',6))
        #             .select(RGB)                        
        #             .median()
        #             )
        # bear        
        # image= (ee.ImageCollection("COPERNICUS/S2_SR")
        #             .filterBounds(lyon.bounds())  
        #             .filter(ee.Filter.calendarRange(2017, 2019,  'year'))
        #             .filter(ee.Filter.calendarRange(6, 10, 'month'))
        #             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',15))
        #             .select(RGB)                        
        #             .median()
            
        # )
        # monkey
        image= (ee.ImageCollection("COPERNICUS/S2_SR")
                    .filterBounds(lyon)  
                    .filterDate('2019-01-15', '2019-12-15')

                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
                    .select(RGB)
                    .median()) 
        minn = self.min
        maxx = 3000
        
        url = image.getThumbURL({
              'bands': RGB,
              'min': [minn, minn, minn], 
              'max': [maxx, maxx, maxx],
              'gamma': 1,
              'region': lyon.bounds(),
              'crs': self.crs,
              'dimensions': self.dimensions,
              'bestEffort': True,
              'format': 'png'})
                                
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            print("Failed to get image...")
            raise r.raise_for_status()
        
        filename = self.sat_path  +'/%05d.png' % index
        with open(filename, 'wb') as out_file:
            shutil.copyfileobj(r.raw, out_file)
        print(f"Done: {filename}")       

    def workSAT(self,indexes, points):
        pool = multiprocessing.Pool()
        pool.starmap(self.getSAT, zip(indexes, points))
        pool.close()
        pool.join()


