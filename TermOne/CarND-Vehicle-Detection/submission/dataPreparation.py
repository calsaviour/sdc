import glob
import matplotlib.image as mpimg

def addImages(images):
    results = []
    for image in images:
        results.append(image)
    return results

def getCars():
	cars = []
	carFar = glob.glob("../dataset/vehicles/GTI_Far/*.png")
	carLeft = glob.glob("../dataset/vehicles/GTI_Left/*.png")
	carMiddleClose = glob.glob("../dataset/vehicles/GTI_MiddleClose/*.png")
	carRight = glob.glob("../dataset/vehicles/GTI_Right/*.png")
	carKitti = glob.glob("../dataset/vehicles/KITTI_extracted/*.png")
	cars = addImages(carFar) + addImages(carLeft) + addImages(carMiddleClose) + addImages(carRight) + addImages(carKitti)
	return cars

def getNonCars():
	notcars = []
	notCarImagesExtras = glob.glob("../dataset/non-vehicles/Extras/*png")
	notCarImagesExtrasGTI = glob.glob("../dataset/non-vehicles/GTI/*png")
	notcars = addImages(notCarImagesExtras) + addImages(notCarImagesExtrasGTI)
	return notcars

# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(notcar_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict