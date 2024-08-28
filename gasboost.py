from pymongo import MongoClient
import streamlit as st
import pandas as pd
import openrouteservice as ors
import math
import googlemaps
from dotenv import load_dotenv
import os

load_dotenv()

ors_key = os.getenv("ORS_API_KEY")
gmaps_key = os.getenv("GMAPS_API_KEY")
mongo_uri = os.getenv("MONGO_URI")

clientDB = MongoClient(mongo_uri)
db = clientDB["Travelboost"]
collection = db["gas_stations"]

client =ors.Client(key=ors_key)
def geocode_address(address, gmaps_client):
    """Geocode an address into latitude and longitude."""
    geocode_result = gmaps_client.geocode(address)
    if geocode_result:
        lat_lng = geocode_result[0]['geometry']['location']
        return (lat_lng['lat'], lat_lng['lng'])
    return None
def get_public_transportation_info(start, end):
    gmaps = googlemaps.Client(key=gmaps_key)

    # Ensure start and end are strings
    start = str(start)
    end = str(end)
    
    # Debugging output
    st.write(f"Fetching directions from '{start}' to '{end}'...")
    # API call to get directions with public transport
    directions = gmaps.directions(
        origin=start,
        destination=end,
        mode="transit"
    )

    return directions

def get_road_points(adresse_coord, dest_coord):
    coords = [adresse_coord, dest_coord]
    route = client.directions(
        coordinates=coords,
        profile='driving-car',
        format='geojson',
        maneuvers=True
    )
    
    steps = route['features'][0]['properties']['segments'][0]['steps']
    geometry = route['features'][0]['geometry']['coordinates']
    total_distance = route['features'][0]['properties']['segments'][0]['distance']
    
    road_points = []
    names = []
    
    # Extracting maneuver points based on distance criteria
    for step in steps:
        if step['distance'] > 1000:
            road_points.append(step['maneuver']['location'])
            names.append(step['name'])
    
    # If maneuver points are sparse, add points every 5 km
    if total_distance / 1000 > len(road_points) * 5:
        cumulative_distance = 0
        interval = 5000  # 5 kilometers in meters
        distances = [0]
        
        # Calculate distances between consecutive geometry points
        for i in range(1, len(geometry)):
            prev_point = geometry[i - 1]
            curr_point = geometry[i]
            
            # Haversine formula to calculate distance between two lat-long points
            distance = haversine_distance(prev_point, curr_point)
            cumulative_distance += distance
            distances.append(cumulative_distance)
        
        # Extract points at every 5 km interval
        next_interval = interval
        for i, dist in enumerate(distances):
            if dist >= next_interval:
                road_points.append(geometry[i])
                names.append(f"Point at {dist/1000:.1f} km")
                next_interval += interval
                
    return road_points, names

def haversine_distance(point1, point2):
    

    # Earth radius in meters
    R = 6371000
    lat1, lon1 = math.radians(point1[1]), math.radians(point1[0])
    lat2, lon2 = math.radians(point2[1]), math.radians(point2[0])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def adresses_geo2(adresses):
    coords = []
    for adresse in adresses:
        geocode = client.pelias_search(text=adresse)
        place = geocode['features'][0]['geometry']['coordinates']
        coords.append(place)
    return coords

def get_all_road_points_with_cut(list_coords):
    all_road_points_with_start_end = []

    # Iterate through each pair of adjacent addresses
    for i in range(len(list_coords) - 1):
        start_coord = list_coords[i]
        end_coord = list_coords[i + 1]
        
        # Get road points between the current pair of coordinates
        road_points, names = get_road_points(start_coord, end_coord)
        
        # Create a list with start, road points, and end
        segment_points = [start_coord] + road_points + [end_coord]
        
        # Append to the all_road_points_with_start_end list
        all_road_points_with_start_end.extend(segment_points)
    
    # Remove duplicates while preserving order
    seen = set()
    all_road_points_no_duplicates = []
    for point in all_road_points_with_start_end:
        if tuple(point) not in seen:
            seen.add(tuple(point))
            all_road_points_no_duplicates.append(point)
    
    
    
    
    return all_road_points_no_duplicates

# Function to find nearby gas stations of a specific fuel type
def find_nearby_gas_stations(road_point, fuel_type, max_distance=5000):
    return list(collection.find({
        "geolocation": {
            "$near": {
                "$geometry": {
                    "type": "Point",
                    "coordinates": road_point
                },
                "$maxDistance": max_distance  # Distance in meters
            }
        },
        "prix_nom": fuel_type  # Filter by fuel type
    }))

def points_cut_all(liste) :
    n=len(liste)
    r=n%6
    geos= liste[:n-r]
    geo_l=liste[n-r::]
    nombre_list= int(len(geos)/6)
    k=1 ;j=0
    list_geo=[]
    for i in range(nombre_list):
        geo=geos[j:6*k]
        j=j+6 ; k=k+1
        list_geo.append(geo)
    list_geo.append(geo_l)
    return list_geo 

def calcul_matrix_distance(l, mode, sources, destinations, metrics):

    result = client.distance_matrix(l, profile=mode, sources=sources, destinations=destinations,units="km", metrics= metrics)

    return result

def calcule_distance(points_geo, station_geo, mode):
    distances = []
    durations = []
    for i in range(len(points_geo)):
        # Concaténer points_geo et station_geo
        geo = points_geo[i] + station_geo
    
        # Initialiser source et destination
        len_source = len(points_geo[i])
        source = list(range(len_source))
        destinations = list(range(len_source, len(geo)))
    
        # Appeler la fonction calcul_matrix_distance
        matrice = calcul_matrix_distance(geo, mode, source, destinations, metrics=["distance", "duration"])
    
        # Extraire les distances et les durées de la matrice
        distances_ = matrice['distances']
        durations_ = matrice['durations']  
    
        # Ajouter les distances et les durées aux listes respectives
        distances += distances_
        durations += durations_
    
    # Structurer les distances et les durées dans des DataFrames
    df_distances = pd.DataFrame(distances)
    df_durations = pd.DataFrame(durations) / 60  # Convertir les durées en minutes
    
    # Retourner un tuple contenant les deux DataFrames
    return df_distances.T, df_durations.T

def find_pareto_front(df):
    # Initialize a boolean array for dominated points
    is_dominated = pd.Series([False] * len(df), index=df.index)

    # Extract the Distance and Price columns for easier comparison
    distance = df['Distance'].values
    price = df['Price'].values

    # Loop through each point to check for domination
    for i in range(len(df)):
        # Create a boolean mask to identify points that dominate the current point
        mask = (distance <= distance[i]) & (price <= price[i]) & ((distance < distance[i]) | (price < price[i]))
        
        if mask.sum() > 0:  # If any point dominates the current point
            is_dominated[i] = True

    # Filter out the dominated rows
    pareto_front = df[~is_dominated]
    
    return pareto_front

# Function to add more address input fields
def add_address_fields():
    if "addresses" not in st.session_state:
        st.session_state.addresses = [1]

    add_button = st.button("Add another address")
    if add_button:
        st.session_state.addresses.append(len(st.session_state.addresses) + 1)

    for i in range(len(st.session_state.addresses)):
        st.text_input(f"Address {i + 1}", key=f"address_{i}")

def calculer_score_pareto(df_pareto, ratio_tolerance):
    """
    Calcule un score pour chaque station dans le front de Pareto, basé sur un ratio de tolérance au détour par rapport au gain de coût.
    
    :param df_pareto: DataFrame contenant les stations optimales avec les colonnes 'duree_detour' et 'cout_revient'.
    :param ratio_tolerance: Le ratio tolérance-coût en minutes par euro (ex: 0.15 pour 15 minutes par 100 euros).
    
    :return: Un DataFrame avec une colonne supplémentaire 'score' représentant le score de chaque station.
    """
    # Calculer la réduction de coût par rapport au coût minimal dans le front Pareto
    cout_minimal = df_pareto['Price_total'].min()
    df_pareto['reduction_cout'] = cout_minimal - df_pareto['Price_total']
    
    # Calculer le score en fonction de la durée du détour et de la réduction de coût
    df_pareto['score'] = df_pareto['Duration'] - ratio_tolerance * df_pareto['reduction_cout']
    
    return df_pareto





def main():
    st.title("Route Optimization App")

    # Sidebar navigation
    page = st.sidebar.radio("Choose your action", ["Gas Station Finder", "Public Transportation"])
    # Call the function to display address fields
    add_address_fields()
    

    if page == "Gas Station Finder":
        
        gas_station_finder()
    elif page == "Public Transportation":
        public_transportation_info()

def gas_station_finder():
    # Display the phrase
    st.title("FuelOptimizer :Find your optimal gas station!")
    if st.button("Show addresses"):
       addresses = [st.session_state[f"address_{i}"] for i in range(len(st.session_state.addresses))]
       st.write("You have entered the following addresses:")
       st.write(addresses)

    # Your existing code for finding the optimal gas station
    st.write("Gas station finder functionality goes here.")
    # Vehicle Type Selection
    vehicle_type = st.selectbox("Type de véhicule", ["Voiture de taille moyenne", "Petite voiture"])

    # Set default values based on vehicle type and fuel type
    if vehicle_type == "Voiture de taille moyenne":
        fuel_type = st.selectbox("Carburant", ["Gazole", "E10"])
        if fuel_type == "Gazole":
            price_per_liter = st.number_input("Prix du litre (€)", value=1.673)
            avg_consumption = st.number_input("Consommation moyenne (L/100km)", value=5.2)
        elif fuel_type == "E10":
            price_per_liter = st.number_input("Prix du litre (€)", value=1.778)
            avg_consumption = st.number_input("Consommation moyenne (L/100km)", value=7.5)
    elif vehicle_type == "Petite voiture":
        fuel_type = st.selectbox("Carburant", ["Gazole", "E10"])
        if fuel_type == "Gazole":
            price_per_liter = st.number_input("Prix du litre (€)", value=1.673)
            avg_consumption = st.number_input("Consommation moyenne (L/100km)", value=5.3)
        elif fuel_type == "E10":
            price_per_liter = st.number_input("Prix du litre (€)", value=1.778)
            avg_consumption = st.number_input("Consommation moyenne (L/100km)", value=7.2)
    tank_capacity = st.number_input("Capacité du réservoir (L)", value=50)



    list_places_cord=adresses_geo2(addresses)
    full_list_geo = get_all_road_points_with_cut(list_places_cord)

    # Aggregate results from all road points
    all_nearby_stations = []
    for point in full_list_geo:
        nearby_stations = find_nearby_gas_stations(point, fuel_type)
        all_nearby_stations.extend(nearby_stations)

    # Optional: Remove duplicates based on a unique identifier (e.g., station ID)
    unique_stations = {station['_id']: station for station in all_nearby_stations}
    coordinates_list = []

    for station in unique_stations.values():
        # Extract the coordinates from the geolocation field
        coordinates = station['geolocation']['coordinates']
        # Append the coordinates to the list
        coordinates_list.append(coordinates)



    full_list_geo_cut = points_cut_all(full_list_geo)


    mode = "driving-car"
    dis,dur =calcule_distance(full_list_geo_cut, coordinates_list,mode)

    price = []

    for station in unique_stations.values():
        # Extract relevant information
        prix_valeur = station['prix_valeur']
        adresse = station['adresse']
        code_postal = station['cp']
        
        # Append to the data list
        price.append({
            'prix_valeur': prix_valeur,
            'adresse': adresse,
            'code_postal': code_postal
        })

    # Create a DataFrame
    pricedf = pd.DataFrame(price)

    minimized_distances = dis.min(axis=1)
    minimized_durations = dur.min(axis=1)

    # Combining distances and prices into a single dataframe
    dfc = pd.DataFrame({
        'Distance': minimized_distances,
        'Duration': minimized_durations,
        'Price': pricedf['prix_valeur'],
        'Address': pricedf['adresse'],
        'Postal Code': pricedf['code_postal']
    })



    pareto_front_df = find_pareto_front(dfc)
    pareto_front_df['Distance_price'] = pareto_front_df['Distance']*avg_consumption*price_per_liter/100 #5L- 100km, 1.5e - 1l donc distance*5*1.5
    pareto_front_df['Price_total'] = pareto_front_df['Price']*tank_capacity + pareto_front_df['Distance_price']


    ratio_tolerance = 0.15
    pareto_front_scored = calculer_score_pareto(pareto_front_df, ratio_tolerance)

    st.subheader("Pareto Front Solutions for Gas Stations")
    st.dataframe(pareto_front_scored)

    # Find the solution with the minimum distance
    min_distance_solution = pareto_front_scored.loc[pareto_front_df['Distance'].idxmin()]

    # Find the solution with the minimum Price_total
    min_price_solution = pareto_front_scored.loc[pareto_front_df['Price_total'].idxmin()]

    # Display the gas station with the shortest distance
    st.write("The gas station with the shortest distance to the path is:")
    st.write(f"**Address**: {min_distance_solution['Address']}, {min_distance_solution['Postal Code']}")
    st.write(f"**Distance**: {min_distance_solution['Distance']} km")
    st.write(f"**Total Price**: {min_distance_solution['Price_total']} €")
    st.write(f"**Duration**: {min_distance_solution['Duration']} minutes")                                  

    # Add some spacing or separator
    st.markdown("---")

    # Display the gas station with the minimum price total
    st.write("The gas station with the minimum total price is:")
    st.write(f"**Address**: {min_price_solution['Address']}, {min_price_solution['Postal Code']}")
    st.write(f"**Distance**: {min_price_solution['Distance']} km")
    st.write(f"**Total Price**: {min_price_solution['Price_total']} €")
    st.write(f"**Duration**: {min_price_solution['Duration']} minutes")



    optimal_station = pareto_front_scored.sort_values(by='score').iloc[0]
    st.markdown("---")
    st.write("The optimal gas station based on the score is:")
    st.write(f"**Address**: {optimal_station['Address']}, {optimal_station['Postal Code']}")
    st.write(f"**Distance**: {optimal_station['Distance']} km")
    st.write(f"**Total Price**: {optimal_station['Price_total']} €")
    st.write(f"**Duration**: {optimal_station['Duration']} minutes")

def public_transportation_info():
    st.write("## Public Transportation Information")

    if st.button("Show addresses"):
       addresses = [st.session_state[f"address_{i}"] for i in range(len(st.session_state.addresses))]
       st.write("You have entered the following addresses:")
       st.write(addresses)
    
        
    # Initialize the Google Maps client
    gmaps_client = googlemaps.Client(key=gmaps_key)

    # Geocode each address
    coordinates = []
    for address in addresses:
        lat_lng = geocode_address(address, gmaps_client)
        if lat_lng:
            coordinates.append(lat_lng)
        else:
            st.write(f"Error geocoding address: {address}")
            return

    # Loop through each pair of consecutive geocoded coordinates to get directions
    for i in range(len(coordinates) - 1):
        start_coords = coordinates[i]
        end_coords = coordinates[i + 1]
        st.write(f"**Route from {addresses[i]} to {addresses[i + 1]}:**")

        directions = gmaps_client.directions(
            origin=start_coords,
            destination=end_coords,
            mode="transit"
        )

        if directions:
            leg = directions[0]['legs'][0]
            st.write(f"Duration: {leg['duration']['text']}")
            st.write("Steps:")
            for step in leg['steps']:
                st.write(f"{step['html_instructions']} ({step['duration']['text']})")
                if 'transit_details' in step:
                    transit = step['transit_details']
                    st.write(f"  - Vehicle: {transit['line']['vehicle']['type']}")
                    if 'fare' in transit:
                        st.write(f"  - Price: {transit['fare']['text']}")  # If fare information is available
                    else:
                        st.write(f"  - Price: Not available")
if __name__ == "__main__":
    main()











# # Display a slider for setting the weight for the distance criterion
# distance_weight = st.slider("Select the weight for the distance criterion (%)", 0, 100, 50)

# # Calculate the weight for Price_total
# price_weight = 100 - distance_weight

# # Display the selected weights
# st.write(f"Weight for Distance: **{distance_weight}%**")
# st.write(f"Weight for Price_total: **{price_weight}%**")

# # Step 1: Normalize the Decision Matrix
# pareto_front_df['norm_distance'] = pareto_front_df['Distance'] / np.sqrt((pareto_front_df['Distance'] ** 2).sum())
# pareto_front_df['norm_price'] = pareto_front_df['Price_total'] / np.sqrt((pareto_front_df['Price_total'] ** 2).sum())

# # Step 2: Apply the Weights
# pareto_front_df['weighted_distance'] = pareto_front_df['norm_distance'] * distance_weight
# pareto_front_df['weighted_price'] = pareto_front_df['norm_price'] * price_weight

# # Step 3: Determine Ideal and Negative-Ideal Solutions
# ideal_solution = {
#     'distance': pareto_front_df['weighted_distance'].min(),
#     'price': pareto_front_df['weighted_price'].min()
# }
# negative_ideal_solution = {
#     'distance': pareto_front_df['weighted_distance'].max(),
#     'price': pareto_front_df['weighted_price'].max()
# }

# # Step 4: Calculate the Distance to Ideal and Negative-Ideal Solutions
# pareto_front_df['distance_to_ideal'] = np.sqrt(
#     (pareto_front_df['weighted_distance'] - ideal_solution['distance']) ** 2 +
#     (pareto_front_df['weighted_price'] - ideal_solution['price']) ** 2
# )
# pareto_front_df['distance_to_negative_ideal'] = np.sqrt(
#     (pareto_front_df['weighted_distance'] - negative_ideal_solution['distance']) ** 2 +
#     (pareto_front_df['weighted_price'] - negative_ideal_solution['price']) ** 2
# )

# # Step 5: Calculate the TOPSIS Score
# pareto_front_df['topsis_score'] = pareto_front_df['distance_to_negative_ideal'] / (
#     pareto_front_df['distance_to_ideal'] + pareto_front_df['distance_to_negative_ideal']
# )

# # Step 6: Rank the Alternatives
# pareto_front_df['rank'] = pareto_front_df['topsis_score'].rank(ascending=False)

# # Find the best gas station
# best_station = pareto_front_df.loc[pareto_front_df['topsis_score'].idxmax()]

# # Display the DataFrame with TOPSIS scores and ranks
# st.dataframe(pareto_front_df)

# # Display the best gas station
# st.write("The best gas station according to TOPSIS is:")
# st.write(f"**Address**: {best_station['Address']}, {best_station['Postal Code']}")
# st.write(f"**Distance**: {best_station['Distance']} km")
# st.write(f"**Total Price**: {best_station['Price_total']} €")
# st.write(f"**TOPSIS Score**: {best_station['topsis_score']}")




