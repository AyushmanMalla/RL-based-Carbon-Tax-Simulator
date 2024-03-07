import pandas as pd

class DataHandler:
    def load_data(self):
        self.agriculture = pd.read_csv('simulation_log_Agriculture.csv')
        self.agriculture = self.agriculture.iloc[-1]

        self.logistic = pd.read_csv('simulation_log_Logistic.csv')
        self.logistic = self.logistic.iloc[-1]

        self.manufacturing = pd.read_csv('simulation_log_Manufacturing.csv')
        self.manufacturing = self.manufacturing.iloc[-1]

        self.tech = pd.read_csv('simulation_log_Tech.csv')
        self.tech = self.tech.iloc[-1]

        return self.agriculture, self.logistic, self.manufacturing, self.tech
    
    # sim_ev                   5.000000e+02
    # sim_petrol_car           0.000000e+00
    # sim_solar_panel          1.257000e+03
    # sim_power_generation     2.514000e+03
    # sim_power_consumption    8.000000e+03
    # sim_electricity_price    1.690000e-01
    # sim_carbon_tax           9.000000e-01
    # sim_revenue              5.000000e+05
    # sim_expenses             4.607819e+05
    # sim_money_in_bank        6.833034e+08
    # sim_current_day          5.000000e+03
    # sim_max_car              5.000000e+02
    # sim_max_electricity      1.600000e+04
    # sim_surplus_power            0.000000e+00

    def calculate_car_rate(self):
        self.ev_rate_agriculture = self.agriculture['sim_ev'] / self.agriculture['sim_max_car']
        self.ev_rate_logistic = self.logistic['sim_ev'] / self.logistic['sim_max_car']
        self.ev_rate_manufacturing = self.manufacturing['sim_ev'] / self.manufacturing['sim_max_car']
        self.ev_rate_tech = self.tech['sim_ev'] / self.tech['sim_max_car']

        self.petrol_rate_agriculture = self.agriculture['sim_petrol_car'] / self.agriculture['sim_max_car']
        self.petrol_rate_logistic = self.logistic['sim_petrol_car'] / self.logistic['sim_max_car']
        self.petrol_rate_manufacturing = self.manufacturing['sim_petrol_car'] / self.manufacturing['sim_max_car']
        self.petrol_rate_tech = self.tech['sim_petrol_car'] / self.tech['sim_max_car']

        return self.ev_rate_agriculture, self.ev_rate_logistic, self.ev_rate_manufacturing, self.ev_rate_tech, self.petrol_rate_agriculture, self.petrol_rate_logistic, self.petrol_rate_manufacturing, self.petrol_rate_tech
    
    def calculate_power_rate(self):
        self.solar_rate_agriculture = self.agriculture['sim_power_generation'] / self.agriculture['sim_max_electricity']
        self.solar_rate_logistic = self.logistic['sim_power_generation'] / self.logistic['sim_max_electricity']
        self.solar_rate_manufacturing = self.manufacturing['sim_power_generation'] / self.manufacturing['sim_max_electricity']
        self.solar_rate_tech = self.tech['sim_power_generation'] / self.tech['sim_max_electricity']

        self.black_power_rate_agriculture = (self.agriculture['sim_power_consumption'] - self.agriculture['sim_power_generation']) / self.agriculture['sim_max_electricity']
        self.black_power_rate_logistic = (self.logistic['sim_power_consumption'] - self.logistic['sim_power_generation']) / self.logistic['sim_max_electricity']
        self.black_power_rate_manufacturing = (self.manufacturing['sim_power_consumption'] - self.manufacturing['sim_power_generation']) / self.manufacturing['sim_max_electricity']
        self.black_power_rate_tech = (self.tech['sim_power_consumption'] - self.tech['sim_power_generation']) / self.tech['sim_max_electricity']

        self.surplus_power_agriculture = self.agriculture['sim_surplus_power']
        self.surplus_power_logistic = self.logistic['sim_surplus_power']
        self.surplus_power_manufacturing = self.manufacturing['sim_surplus_power']
        self.surplus_power_tech = self.tech['sim_surplus_power']

        return self.solar_rate_agriculture, self.solar_rate_logistic, self.solar_rate_manufacturing, self.solar_rate_tech, self.black_power_rate_agriculture, self.black_power_rate_logistic, self.black_power_rate_manufacturing, self.black_power_rate_tech, self.surplus_power_agriculture, self.surplus_power_logistic, self.surplus_power_manufacturing, self.surplus_power_tech
    
        


test = DataHandler()
print(test.load_data())