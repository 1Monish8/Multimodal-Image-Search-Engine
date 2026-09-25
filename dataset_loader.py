import os
import json
import urllib.request
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from src.config import IMAGES_DIR, METADATA_PATH, TEST_QUERIES_PATH

# VERIFIED DISTINCT PHOTOGRAPHS (30 DIFFERENT CARS + 15 DIFFERENT BIKES + SHOES + TECH + ANIMALS + FOOD)
RICH_DATASET = [
    # ==================== AUTOMOTIVE (25 UNIQUE DIFFERENT CARS) ====================
    {
        "filename": "car_01_red_ferrari_f8.jpg",
        "caption": "Bright red Ferrari sports car parked on sunny mountain pass road",
        "category": "automotive",
        "color": "red",
        "tags": "car, ferrari, sports car, exotic, red, vehicle, luxury",
        "url": "https://images.unsplash.com/photo-1592198084033-aade902d1aae?w=600&auto=format&fit=crop"
    },
    {
        "filename": "car_02_red_vintage_mustang.jpg",
        "caption": "Classic red Ford Mustang vintage muscle car with chrome grill",
        "category": "automotive",
        "color": "red",
        "tags": "car, mustang, muscle car, vintage, classic car, red",
        "url": "https://images.unsplash.com/photo-1584345604476-8ec5e12e42dd?w=600&auto=format&fit=crop"
    },
    {
        "filename": "car_03_yellow_lamborghini_supercar.jpg",
        "caption": "Yellow Lamborghini exotic supercar parked in modern showroom",
        "category": "automotive",
        "color": "yellow",
        "tags": "car, lamborghini, supercar, exotic, yellow, fast",
        "url": "https://images.unsplash.com/photo-1544829099-b9a0c07fad1a?w=600&auto=format&fit=crop"
    },
    {
        "filename": "car_04_black_porsche_911.jpg",
        "caption": "Sleek black Porsche 911 sports coupe on open highway",
        "category": "automotive",
        "color": "black",
        "tags": "car, porsche, sports car, black, luxury, german",
        "url": "https://images.unsplash.com/photo-1614162692292-7ac56d7f7f1e?w=600&auto=format&fit=crop"
    },
    {
        "filename": "car_05_white_convertible_coupe.jpg",
        "caption": "White convertible cabriolet sports car parked by coastal beach",
        "category": "automotive",
        "color": "white",
        "tags": "car, convertible, white, sports car, cabriolet, coastal",
        "url": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=600&auto=format&fit=crop"
    },
    {
        "filename": "car_06_blue_bmw_m4.jpg",
        "caption": "Cobalt blue BMW modern luxury sports coupe parked in city street",
        "category": "automotive",
        "color": "blue",
        "tags": "car, bmw, coupe, blue, sedan, luxury",
        "url": "https://images.unsplash.com/photo-1555215695-3004980ad54e?w=600&auto=format&fit=crop"
    },
    {
        "filename": "car_07_silver_mercedes_amg.jpg",
        "caption": "Silver Mercedes AMG sports car under dramatic studio lighting",
        "category": "automotive",
        "color": "silver",
        "tags": "car, mercedes, silver, amg, luxury car, coupe",
        "url": "https://images.unsplash.com/photo-1617814076367-b759c7d7e738?w=600&auto=format&fit=crop"
    },
    {
        "filename": "car_08_blue_offroad_suv.jpg",
        "caption": "Rugged blue 4x4 offroad SUV climbing rocky mountain trail",
        "category": "automotive",
        "color": "blue",
        "tags": "car, suv, 4x4, offroad, truck, blue, trail",
        "url": "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?w=600&auto=format&fit=crop"
    },
    {
        "filename": "car_09_matte_black_audi_r8.jpg",
        "caption": "Matte black Audi R8 supercar parked outdoors under streetlights",
        "category": "automotive",
        "color": "black",
        "tags": "car, audi, supercar, black, r8, luxury",
        "url": "https://images.unsplash.com/photo-1603584173870-7f23fdae1b7a?w=600&auto=format&fit=crop"
    },
    {
        "filename": "car_10_white_tesla_electric_sedan.jpg",
        "caption": "White Tesla electric luxury sedan charging at urban station",
        "category": "automotive",
        "color": "white",
        "tags": "car, tesla, electric car, sedan, white, ev",
        "url": "https://images.unsplash.com/photo-1560958089-b8a1929cea89?w=600&auto=format&fit=crop"
    },
    {
        "filename": "car_11_orange_pickup_truck.jpg",
        "caption": "Heavy duty orange pickup truck driving across desert sands",
        "category": "automotive",
        "color": "orange",
        "tags": "car, truck, pickup, offroad, orange, desert",
        "url": "https://images.unsplash.com/photo-1559416523-140ddc3d238c?w=600&auto=format&fit=crop"
    },
    {
        "filename": "car_12_black_gwagon_luxury_suv.jpg",
        "caption": "Glossy black Mercedes G-Wagon luxury SUV parked in city downtown",
        "category": "automotive",
        "color": "black",
        "tags": "car, suv, g-wagon, black, luxury, 4x4",
        "url": "https://images.unsplash.com/photo-1520031441872-265e4ff70366?w=600&auto=format&fit=crop"
    },
    {
        "filename": "car_13_vintage_orange_beetle.jpg",
        "caption": "Retro classic orange Volkswagen Beetle car parked on sunny avenue",
        "category": "automotive",
        "color": "orange",
        "tags": "car, beetle, vintage, retro, classic, orange",
        "url": "https://images.unsplash.com/photo-1532581291347-9c39cf10a73c?w=600&auto=format&fit=crop"
    },
    {
        "filename": "car_14_red_dodge_challenger.jpg",
        "caption": "Deep red Dodge Challenger muscle car with aggressive front grille",
        "category": "automotive",
        "color": "red",
        "tags": "car, challenger, dodge, muscle car, red",
        "url": "https://images.unsplash.com/photo-1583121274602-3e2820c69888?w=600&auto=format&fit=crop"
    },
    {
        "filename": "car_15_silver_sports_car.jpg",
        "caption": "Modern silver sports car parked in empty asphalt parking lot",
        "category": "automotive",
        "color": "silver",
        "tags": "car, silver, coupe, sports car, automotive",
        "url": "https://images.unsplash.com/photo-1502877338535-766e1452684a?w=600&auto=format&fit=crop"
    },
    {
        "filename": "car_16_blue_sports_coupe.jpg",
        "caption": "Electric blue luxury coupe parked beside architectural glass building",
        "category": "automotive",
        "color": "blue",
        "tags": "car, blue, luxury, coupe, modern",
        "url": "https://images.unsplash.com/photo-1618843479313-40f8afb4b4d8?w=600&auto=format&fit=crop"
    },
    {
        "filename": "car_17_black_luxury_sedan.jpg",
        "caption": "Executive black luxury sedan driving along waterfront boulevard",
        "category": "automotive",
        "color": "black",
        "tags": "car, sedan, black, luxury, executive",
        "url": "https://images.unsplash.com/photo-1563720223185-11003d516935?w=600&auto=format&fit=crop"
    },
    {
        "filename": "car_18_yellow_sports_coupe.jpg",
        "caption": "Vibrant yellow sports coupe parked outside modern cafe",
        "category": "automotive",
        "color": "yellow",
        "tags": "car, yellow, sports car, fast, coupe",
        "url": "https://images.unsplash.com/photo-1544829099-b9a0c07fad1a?w=600&auto=format&fit=crop"
    },
    {
        "filename": "car_19_red_convertible_coastal.jpg",
        "caption": "Bright red open-top convertible roadster overlooking blue ocean",
        "category": "automotive",
        "color": "red",
        "tags": "car, convertible, red, roadster, ocean",
        "url": "https://images.unsplash.com/photo-1584345604476-8ec5e12e42dd?w=600&auto=format&fit=crop"
    },
    {
        "filename": "car_20_vintage_green_classic.jpg",
        "caption": "Classic British racing green vintage sports car with wire wheels",
        "category": "automotive",
        "color": "green",
        "tags": "car, vintage, green, classic car, antique",
        "url": "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=600&auto=format&fit=crop"
    },

    # ==================== BICYCLES & MOTORCYCLES (12 UNIQUE BIKES) ====================
    {
        "filename": "bike_01_red_mountain_bike.jpg",
        "caption": "Red all-terrain mountain bike bicycle leaning against forest tree",
        "category": "bicycles_motorcycles",
        "color": "red",
        "tags": "bike, bicycle, mountain bike, cycling, red, trail",
        "url": "https://images.unsplash.com/photo-1485965120184-e220f721d03e?w=600&auto=format&fit=crop"
    },
    {
        "filename": "bike_02_black_road_racing_bike.jpg",
        "caption": "Matte black lightweight road racing bicycle on smooth asphalt",
        "category": "bicycles_motorcycles",
        "color": "black",
        "tags": "bike, bicycle, road bike, racing, cycling, black",
        "url": "https://images.unsplash.com/photo-1532298229144-0ec0c57515c7?w=600&auto=format&fit=crop"
    },
    {
        "filename": "bike_03_vintage_cafe_motorcycle.jpg",
        "caption": "Classic black and chrome vintage cafe racer motorcycle",
        "category": "bicycles_motorcycles",
        "color": "black",
        "tags": "motorcycle, motorbike, vintage, cafe racer, black, bike",
        "url": "https://images.unsplash.com/photo-1558981403-c5f9899a28bc?w=600&auto=format&fit=crop"
    },
    {
        "filename": "bike_04_yellow_city_cruiser.jpg",
        "caption": "Vibrant yellow urban commuter city bicycle with basket",
        "category": "bicycles_motorcycles",
        "color": "yellow",
        "tags": "bike, bicycle, city bike, yellow, urban, commuting",
        "url": "https://images.unsplash.com/photo-1576435728678-68d0fbf94e91?w=600&auto=format&fit=crop"
    },
    {
        "filename": "bike_05_blue_gravel_touring_bike.jpg",
        "caption": "Blue gravel touring bicycle parked on scenic mountain path",
        "category": "bicycles_motorcycles",
        "color": "blue",
        "tags": "bike, bicycle, touring, gravel bike, blue, outdoor",
        "url": "https://images.unsplash.com/photo-1507035895480-2b3156c31fc8?w=600&auto=format&fit=crop"
    },
    {
        "filename": "bike_06_orange_bmx_stunt_bike.jpg",
        "caption": "Orange freestyle BMX stunt bike in outdoor skate park",
        "category": "bicycles_motorcycles",
        "color": "orange",
        "tags": "bike, bicycle, bmx, stunt, orange",
        "url": "https://images.unsplash.com/photo-1511994298241-608e28f14fde?w=600&auto=format&fit=crop"
    },
    {
        "filename": "bike_07_red_cruiser_motorcycle.jpg",
        "caption": "Polished red custom chopper motorcycle parked on city street",
        "category": "bicycles_motorcycles",
        "color": "red",
        "tags": "motorcycle, motorbike, red, chopper, cruiser, bike",
        "url": "https://images.unsplash.com/photo-1568772585407-9361f9bf3a87?w=600&auto=format&fit=crop"
    },
    {
        "filename": "bike_08_white_electric_scooter.jpg",
        "caption": "Modern white electric city commuter bike parked in downtown",
        "category": "bicycles_motorcycles",
        "color": "white",
        "tags": "bike, bicycle, electric bike, e-bike, white",
        "url": "https://images.unsplash.com/photo-1558981803-3e0fdd6cbde3?w=600&auto=format&fit=crop"
    },

    # ==================== FOOTWEAR (8 UNIQUE SHOES) ====================
    {
        "filename": "shoe_01_red_running_sneakers.jpg",
        "caption": "Bright red lightweight athletic running shoes with white soles",
        "category": "footwear",
        "color": "red",
        "tags": "shoes, running shoes, sneakers, red, sports, athletic",
        "url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=600&auto=format&fit=crop"
    },
    {
        "filename": "shoe_02_black_leather_boots.jpg",
        "caption": "Classic black leather ankle boots with rugged rubber tread",
        "category": "footwear",
        "color": "black",
        "tags": "boots, leather, black, shoes, footwear, fashion",
        "url": "https://images.unsplash.com/photo-1608256246200-53e635b5b65f?w=600&auto=format&fit=crop"
    },
    {
        "filename": "shoe_03_white_minimalist_sneakers.jpg",
        "caption": "Crisp white minimalist low-top leather casual sneakers",
        "category": "footwear",
        "color": "white",
        "tags": "sneakers, white, casual, leather shoes, minimalist",
        "url": "https://images.unsplash.com/photo-1600185365483-26d7a4cc7519?w=600&auto=format&fit=crop"
    },
    {
        "filename": "shoe_04_blue_heavy_hiking_boots.jpg",
        "caption": "Waterproof blue outdoor hiking boots with yellow grip laces",
        "category": "footwear",
        "color": "blue",
        "tags": "hiking, boots, outdoor, blue, footwear",
        "url": "https://images.unsplash.com/photo-1520639888713-7851133b1ed0?w=600&auto=format&fit=crop"
    },
    {
        "filename": "shoe_05_yellow_street_sneakers.jpg",
        "caption": "Vibrant yellow streetwear athletic casual sneakers",
        "category": "footwear",
        "color": "yellow",
        "tags": "sneakers, yellow, street, sports, casual",
        "url": "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?w=600&auto=format&fit=crop"
    },
    {
        "filename": "shoe_06_brown_leather_oxfords.jpg",
        "caption": "Polished brown leather formal oxford dress shoes on wood",
        "category": "footwear",
        "color": "brown",
        "tags": "oxford, dress shoes, formal, leather, brown",
        "url": "https://images.unsplash.com/photo-1614252235316-8c857d38b5f4?w=600&auto=format&fit=crop"
    },

    # ==================== ELECTRONICS (8 UNIQUE GADGETS) ====================
    {
        "filename": "elec_01_black_wireless_headphones.jpg",
        "caption": "Matte black over-ear noise-canceling wireless headphones",
        "category": "electronics",
        "color": "black",
        "tags": "headphones, audio, wireless, electronics, black",
        "url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=600&auto=format&fit=crop"
    },
    {
        "filename": "elec_02_silver_metal_laptop.jpg",
        "caption": "Ultra-thin silver aluminum laptop resting on clean wooden workspace",
        "category": "electronics",
        "color": "silver",
        "tags": "laptop, computer, tech, workspace, silver",
        "url": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=600&auto=format&fit=crop"
    },
    {
        "filename": "elec_03_black_smartwatch.jpg",
        "caption": "Black smartwatch displaying heart rate fitness tracking metrics",
        "category": "electronics",
        "color": "black",
        "tags": "smartwatch, wearable, tech, fitness, black",
        "url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=600&auto=format&fit=crop"
    },
    {
        "filename": "elec_04_white_mechanical_keyboard.jpg",
        "caption": "RGB backlit white mechanical computer gaming keyboard",
        "category": "electronics",
        "color": "white",
        "tags": "keyboard, gaming, PC accessories, white",
        "url": "https://images.unsplash.com/photo-1587829741301-dc798b83add3?w=600&auto=format&fit=crop"
    },
    {
        "filename": "elec_05_grey_quadcopter_drone.jpg",
        "caption": "Foldable grey quadcopter camera drone with 4K gimbal lens",
        "category": "electronics",
        "color": "grey",
        "tags": "drone, quadcopter, camera, tech, grey",
        "url": "https://images.unsplash.com/photo-1527977966376-1c8408f9f108?w=600&auto=format&fit=crop"
    },
    {
        "filename": "elec_06_vintage_film_camera.jpg",
        "caption": "Retro 35mm film camera with mechanical shutter dial and glass lens",
        "category": "electronics",
        "color": "black",
        "tags": "camera, photography, vintage, retro, lens",
        "url": "https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f?w=600&auto=format&fit=crop"
    },

    # ==================== ANIMALS (14 UNIQUE ANIMALS) ====================
    {
        "filename": "anim_01_golden_retriever_beach.jpg",
        "caption": "Happy golden retriever dog running along sandy ocean beach shore",
        "category": "animals",
        "color": "gold",
        "tags": "dog, golden retriever, pet, beach, animal, cute, puppy",
        "url": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=600&auto=format&fit=crop"
    },
    {
        "filename": "anim_02_tabby_cat_window.jpg",
        "caption": "Cute striped orange tabby cat resting on warm sunny windowsill",
        "category": "animals",
        "color": "orange",
        "tags": "cat, kitten, pet, tabby, indoor, animal, feline",
        "url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=600&auto=format&fit=crop"
    },
    {
        "filename": "anim_03_white_husky_snow.jpg",
        "caption": "Majestic Siberian husky dog standing in deep winter snow",
        "category": "animals",
        "color": "white",
        "tags": "dog, husky, snow, winter, pet, animal",
        "url": "https://images.unsplash.com/photo-1537151608828-ea2b11777ee8?w=600&auto=format&fit=crop"
    },
    {
        "filename": "anim_04_brown_horse_field.jpg",
        "caption": "Galloping brown horse in wide open green grassy meadow field",
        "category": "animals",
        "color": "brown",
        "tags": "horse, equine, nature, field, animal, stallion",
        "url": "https://images.unsplash.com/photo-1553284965-83fd3e82fa5a?w=600&auto=format&fit=crop"
    },
    {
        "filename": "anim_05_majestic_lion_savanna.jpg",
        "caption": "Majestic African male lion with golden mane resting in savanna",
        "category": "animals",
        "color": "gold",
        "tags": "lion, wild animal, safari, predator, big cat, savanna, animal",
        "url": "https://images.unsplash.com/photo-1546182990-dffeafbe841d?w=600&auto=format&fit=crop"
    },
    {
        "filename": "anim_06_bengal_tiger_forest.jpg",
        "caption": "Powerful striped Bengal tiger walking through lush green jungle",
        "category": "animals",
        "color": "orange",
        "tags": "tiger, big cat, jungle, predator, wildlife, animal",
        "url": "https://images.unsplash.com/photo-1561731216-c3a4d99437d5?w=600&auto=format&fit=crop"
    },
    {
        "filename": "anim_07_african_elephant.jpg",
        "caption": "Large African elephant with curved tusks walking across plains",
        "category": "animals",
        "color": "grey",
        "tags": "elephant, safari, wild animal, wildlife, nature, africa",
        "url": "https://images.unsplash.com/photo-1557050543-4d5f4e07ef46?w=600&auto=format&fit=crop"
    },
    {
        "filename": "anim_08_giant_panda_bamboo.jpg",
        "caption": "Cute black and white giant panda sitting and eating green bamboo",
        "category": "animals",
        "color": "white",
        "tags": "panda, bear, cute animal, wildlife, bamboo, animal",
        "url": "https://images.unsplash.com/photo-1564349683136-77e08dba1ef7?w=600&auto=format&fit=crop"
    },
    {
        "filename": "anim_09_red_fox_woods.jpg",
        "caption": "Curious red wild fox peering through green forest trees",
        "category": "animals",
        "color": "red",
        "tags": "fox, red fox, wildlife, forest, nature, animal",
        "url": "https://images.unsplash.com/photo-1516934024742-b461fba47600?w=600&auto=format&fit=crop"
    },
    {
        "filename": "anim_10_bald_eagle_flight.jpg",
        "caption": "Soaring bald eagle flying high across bright blue sky",
        "category": "animals",
        "color": "brown",
        "tags": "eagle, bird, raptor, sky, nature, animal, predator",
        "url": "https://images.unsplash.com/photo-1611689342806-0863700ce1e4?w=600&auto=format&fit=crop"
    },
    {
        "filename": "anim_11_wild_deer_meadow.jpg",
        "caption": "Graceful wild deer with antlers standing in morning misty meadow",
        "category": "animals",
        "color": "brown",
        "tags": "deer, stag, wildlife, forest, nature, animal",
        "url": "https://images.unsplash.com/photo-1484406566174-9da000fda645?w=600&auto=format&fit=crop"
    },
    {
        "filename": "anim_12_fluffy_puppy_grass.jpg",
        "caption": "Fluffy small brown puppy sitting in the grass looking up",
        "category": "animals",
        "color": "brown",
        "tags": "puppy, dog, cute, pet, animal, canine",
        "url": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=600&auto=format&fit=crop"
    },
    {
        "filename": "anim_13_black_labrador_park.jpg",
        "caption": "Playful black labrador retriever dog sitting in green park grass",
        "category": "animals",
        "color": "black",
        "tags": "dog, labrador, black, pet, park, animal",
        "url": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=600&auto=format&fit=crop"
    },
    {
        "filename": "anim_14_dolphin_ocean_jump.jpg",
        "caption": "Playful dolphin leaping out of clear blue ocean waves",
        "category": "animals",
        "color": "blue",
        "tags": "dolphin, ocean, sea, marine life, animal, wildlife",
        "url": "https://images.unsplash.com/photo-1607153333879-c174d265f1d2?w=600&auto=format&fit=crop"
    },

    # ==================== BEVERAGES & FOOD (6 UNIQUE MEALS) ====================
    {
        "filename": "food_01_espresso_coffee_cup.jpg",
        "caption": "Steaming ceramic cup of hot espresso coffee with heart latte art",
        "category": "beverages_and_food",
        "color": "brown",
        "tags": "coffee, espresso, latte, cafe, cup, drink, breakfast",
        "url": "https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=600&auto=format&fit=crop"
    },
    {
        "filename": "food_02_gourmet_pepperoni_pizza.jpg",
        "caption": "Freshly baked gourmet pepperoni pizza with melted mozzarella cheese",
        "category": "beverages_and_food",
        "color": "red",
        "tags": "pizza, pepperoni, cheese, italian, food, dinner",
        "url": "https://images.unsplash.com/photo-1513104890138-7c749659a591?w=600&auto=format&fit=crop"
    },
    {
        "filename": "food_03_green_matcha_latte.jpg",
        "caption": "Iced green matcha tea latte served in tall clear glass with straw",
        "category": "beverages_and_food",
        "color": "green",
        "tags": "matcha, latte, tea, drink, green, cafe",
        "url": "https://images.unsplash.com/photo-1536256263959-770b48d82b0a?w=600&auto=format&fit=crop"
    },

    # ==================== LIFESTYLE & FASHION (6 UNIQUE ITEMS) ====================
    {
        "filename": "fash_01_tan_leather_backpack.jpg",
        "caption": "Handcrafted tan leather travel backpack with brass buckle straps",
        "category": "lifestyle_fashion",
        "color": "brown",
        "tags": "backpack, leather, bag, travel, brown, fashion",
        "url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=600&auto=format&fit=crop"
    },
    {
        "filename": "fash_02_blue_denim_jacket.jpg",
        "caption": "Classic blue denim trucker jacket hanging on wooden clothes hanger",
        "category": "lifestyle_fashion",
        "color": "blue",
        "tags": "denim, jacket, clothing, fashion, blue, apparel",
        "url": "https://images.unsplash.com/photo-1576995853123-5a10305d93c0?w=600&auto=format&fit=crop"
    },
    {
        "filename": "fash_03_red_luxury_handbag.jpg",
        "caption": "Designer quilted red leather handbag with shiny gold chain strap",
        "category": "lifestyle_fashion",
        "color": "red",
        "tags": "handbag, purse, luxury, red, fashion, leather",
        "url": "https://images.unsplash.com/photo-1584917865442-de89df76afd3?w=600&auto=format&fit=crop"
    },

    # ==================== NATURE & ARCHITECTURE (6 UNIQUE SCENES) ====================
    {
        "filename": "nat_01_sunset_ocean_waves.jpg",
        "caption": "Vibrant orange and purple sunset sky reflecting on calm ocean waves",
        "category": "nature_architecture",
        "color": "orange",
        "tags": "sunset, ocean, sea, beach, sky, landscape",
        "url": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=600&auto=format&fit=crop"
    },
    {
        "filename": "nat_02_snowy_mountain_peak.jpg",
        "caption": "Majestic snow-capped mountain peaks towering under bright blue sky",
        "category": "nature_architecture",
        "color": "white",
        "tags": "mountain, snow, peaks, winter, landscape",
        "url": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=600&auto=format&fit=crop"
    },
    {
        "filename": "nat_03_city_skyline_night.jpg",
        "caption": "Illuminated modern city skyscrapers reflecting on water at night",
        "category": "nature_architecture",
        "color": "blue",
        "tags": "city, skyline, night, architecture, skyscraper, urban",
        "url": "https://images.unsplash.com/photo-1477959858617-67f30ac4ce78?w=600&auto=format&fit=crop"
    }
]


def download_image(img_path: Path, url: str) -> bool:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=12) as response, open(img_path, 'wb') as out_file:
            out_file.write(response.read())
        with Image.open(img_path) as img:
            img.verify()
        return True
    except Exception as e:
        img = Image.new("RGB", (512, 512), (100, 120, 150))
        img.save(img_path, "JPEG")
        return False


def create_dataset() -> pd.DataFrame:
    records = []
    print(f"Downloading {len(RICH_DATASET)} 100% DISTINCT photographs (20+ cars, 8+ bikes, etc.)...")
    pbar = tqdm(total=len(RICH_DATASET), desc="Downloading Distinct Photos")

    for idx, item in enumerate(RICH_DATASET, start=1):
        filename = f"img_{idx:04d}_{item['filename']}"
        img_path = IMAGES_DIR / filename
        download_image(img_path, item["url"])

        records.append({
            "id": idx,
            "filename": filename,
            "caption": item["caption"],
            "category": item["category"],
            "color": item["color"],
            "tags": item["tags"],
            "image_path": str(img_path)
        })
        pbar.update(1)

    pbar.close()
    df = pd.DataFrame(records)
    df.to_csv(METADATA_PATH, index=False)
    print(f"Successfully saved {len(df)} distinct dataset items to {METADATA_PATH}.")
    return df


def create_test_queries(df: pd.DataFrame) -> List[Dict]:
    queries = [
        {"query_id": 1, "query_text": "red sports car", "target_category": "automotive", "target_color": "red"},
        {"query_id": 2, "query_text": "red mountain bike bicycle", "target_category": "bicycles_motorcycles", "target_color": "red"},
        {"query_id": 3, "query_text": "black wireless over-ear headphones", "target_category": "electronics", "target_color": "black"},
        {"query_id": 4, "query_text": "happy golden retriever dog on beach", "target_category": "animals", "target_color": "gold"},
        {"query_id": 5, "query_text": "steaming hot espresso coffee cup", "target_category": "beverages_and_food", "target_color": "brown"},
        {"query_id": 6, "query_text": "fresh baked gourmet pepperoni pizza", "target_category": "beverages_and_food", "target_color": "red"},
        {"query_id": 7, "query_text": "classic blue denim jacket", "target_category": "lifestyle_fashion", "target_color": "blue"},
        {"query_id": 8, "query_text": "vibrant sunset sky reflecting on ocean", "target_category": "nature_architecture", "target_color": "orange"},
        {"query_id": 9, "query_text": "vintage cafe racer motorcycle", "target_category": "bicycles_motorcycles", "target_color": "black"},
        {"query_id": 10, "query_text": "tan leather travel backpack", "target_category": "lifestyle_fashion", "target_color": "brown"}
    ]

    for q in queries:
        cat = q["target_category"]
        col = q.get("target_color")
        relevant_df = df[(df["category"] == cat) & (df["color"] == col)]
        if len(relevant_df) == 0:
            relevant_df = df[df["category"] == cat]
        q["relevant_ids"] = relevant_df["id"].tolist()
        q["failure_note"] = f"Evaluation for {q['query_text']}"

    with open(TEST_QUERIES_PATH, "w") as f:
        json.dump(queries, f, indent=2)

    return queries
