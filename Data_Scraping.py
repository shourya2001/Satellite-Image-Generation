
# Importing necessary libraries
from selenium import webdriver
from selenium.webdriver.common.by import By
from PIL import Image
import time
from tqdm import tqdm
import sys


# Static variables for the map coordinates, user directory, and location name.
# These should be filled out before running the script.
# sys command input is used to take the map coordinates, user directory, and location name as input ex: 
mapCord = sys.argv[1]
user = sys.argv[2]
loc = sys.argv[3]
# mapCord = "-118.33681%2C34.08500%2C15"
# user = ''
# loc = "UCLA"

# Path to the text file containing dates and corresponding Wayback item numbers
file_path = "data.txt"


# Initialize an empty dictionary to store the data
data_dict = {}

# Open the text file and read its contents line by line
with open(file_path, "r") as file:
    for line in file:
        # Split each line into a date and a number
        date, number = line.strip().split()
        # Add the date and number to the dictionary
        data_dict[date] = int(number)


for date in tqdm(data_dict, desc="Capturing Screenshots"):

    browser = webdriver.Firefox()
    browser.maximize_window()

    # Navigate to the URL
    link = 'https://livingatlas.arcgis.com/wayback/#active='+ str(data_dict[date]) +'&mapCenter='+mapCord

    browser.get(link)

    # Wait for the cookie consent popup to load
    time.sleep(2)

    # Click the "Accept All Cookies" button
    accept_button = browser.find_element(By.ID, "onetrust-accept-btn-handler")
    accept_button.click()

    # Wait for the accept button to process
    time.sleep(2)

    checkbox_div = browser.find_element(By.CSS_SELECTOR, "div.margin-left-half.margin-right-quarter.cursor-pointer")
    checkbox_div.click()

    time.sleep(2)

    # Take a screenshot and save it
    # screenshot_path = 'C:/Users/'+user+'/Downloads/'+loc+'/temp.png'
    screenshot_path = user + '/temp.png'
    browser.save_screenshot(screenshot_path)

    # Define the crop area to exclude any unwanted parts of the screenshot
    # These coordinates are specific to the the Digital Creative Lab Screen at USC
    left, upper, right, lower = 400, 60, 1910, 930

    # Open the full screenshot and crop it to the red boundary area
    img = Image.open(screenshot_path)
    cropped_img = img.crop((left, upper, right, lower))

    # Save the cropped image
    # cropped_img_path = 'C:/Users/'+user+'/Downloads/'+loc+'/' + date + '.png'
    cropped_img_path = user + '/' + date + '.png'
    cropped_img.save(cropped_img_path)

    # Clean up by closing the browser
    browser.quit()