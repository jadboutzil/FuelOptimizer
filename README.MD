# FuelOptimizer

FuelOptimizer is a Python script that optimizes fuel consumption for vehicles. It uses advanced algorithms to find an optimal gas station given a certain route to traverse in France.

## Features

- Gives its user the ability to configure his vehicle
- Provides a Pareto Front of optimal gas station
- Returns the closest station to the route, the cheapest one and the best one according to a score

## Installation

1. Clone the repository: `git clone https://github.com/your-username/gasboost.git`
2. Navigate to the project directory: `cd gasboost`
3. Install the required dependencies: `pip install -r requirements.txt`

## Usage

1. Set up MongoDB Atlas
2. Import prix-carburants-fichier-instantane-test-ods-copie.csv into MongoDB using Compass
3. Create an index with the key as 'geolocation' and the type as '2dsphere'
4. Get your own Open Route Services API Key and set it up
5. Run the script: `streamlit run gasboost.py`
6. Follow the on-screen instructions to input your driving data
7. Receive personalized recommendations for optimizing fuel consumption

## Contributing

Contributions are welcome! If you have any ideas or improvements, feel free to open an issue or submit a pull request.
