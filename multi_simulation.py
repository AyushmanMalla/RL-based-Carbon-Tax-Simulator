from multiprocessing import Process
from simulator import Trainer  
import shutil

# multi_simulation.run(agrilculture_tax_rate, logistic_tax_rate, manufacturing_tax_rate, tech_tax_rate, electricity_price) 

def run_trainer(config):  # Function now takes a config dictionary
    trainer = Trainer(**config)  # Unpack the config dictionary 
    trainer.train()

def run(agrilculture_tax_rate, logistic_tax_rate, manufacturing_tax_rate, tech_tax_rate, electricity_price):
    shutil.rmtree('tensorboard', ignore_errors=True)
    
    set_electricity_price = electricity_price
    

    print("Running simulation with electricity price: ", set_electricity_price, " and carbon tax: ", agrilculture_tax_rate, 'for agriculture')
    print("Running simulation with electricity price: ", set_electricity_price, " and carbon tax: ", logistic_tax_rate, 'for logistic')
    print("Running simulation with electricity price: ", set_electricity_price, " and carbon tax: ", manufacturing_tax_rate, 'for manufacturing')
    print("Running simulation with electricity price: ", set_electricity_price, " and carbon tax: ", tech_tax_rate, 'for tech')
    print('===================================================================================================================')

    configurations = [
        {"env_name": "Logistic", "revenue": 200000, "expenses": 180000, "money_in_bank": int(2e7), 'max_car': 1000, 'ev': 5, 'petrol_car': 200, 
         'power_consumption': 2000, 'power_generation': 0, 'solar_panel': 0, 'max_electricity': 5000, 'electricity_price': set_electricity_price, 'carbon_tax': logistic_tax_rate},  # Config 1
        {"env_name": "Manufacturing", "revenue": 300000, "expenses": 295000, "money_in_bank": int(2e7), 'max_car': 100, 'ev': 0, 'petrol_car': 50, 
         'power_consumption': 5000, 'power_generation': 0, 'solar_panel': 0, 'max_electricity': 10000, 'electricity_price': set_electricity_price, 'carbon_tax': manufacturing_tax_rate},  # Config 2
        {"env_name": "Agriculture", "revenue": 30000, "expenses": 25000, "money_in_bank": int(2e7), 'max_car': 3000, 'ev': 10, 'petrol_car': 2, 
        'power_consumption': 3000, 'power_generation': 0, 'solar_panel': 0, 'max_electricity': 10000, 'electricity_price': set_electricity_price, 'carbon_tax': agrilculture_tax_rate},  # Config 3
        {"env_name": "Tech", "revenue": 500000, "expenses": 450000, "money_in_bank": int(2e7), 'max_car': 500, 'ev': 100, 'petrol_car': 0, 
        'power_consumption': 8000, 'power_generation': 0, 'solar_panel': 0, 'max_electricity': 16000, 'electricity_price': set_electricity_price, 'carbon_tax': tech_tax_rate},  # Config 4
    ]
    
    # Create and start processes
    processes = []
    for config in configurations:
        process = Process(target=run_trainer, args=(config,)) # Send config dict 
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

if __name__ == '__main__':
    # remove folder tensorboard
    
    shutil.rmtree('tensorboard', ignore_errors=True)
    run(1, 1)  # Run the function with the desired parameters