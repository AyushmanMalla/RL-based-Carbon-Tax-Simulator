from flask import Flask, render_template, request, jsonify
import multi_simulation

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    data = request.get_json()
    agrilculture_tax_rate = data.get('taxRate_agrilculture')
    logistic_tax_rate = data.get('taxRate_logistic')
    manufacturing_tax_rate = data.get('taxRate_manufacturing')
    tech_tax_rate = data.get('taxRate_tech')
    electricity_price = data.get('electricityPrice')

    agrilculture_tax_rate = float(agrilculture_tax_rate)/1000
    logistic_tax_rate = float(logistic_tax_rate)/1000
    manufacturing_tax_rate = float(manufacturing_tax_rate)/1000
    tech_tax_rate = float(tech_tax_rate)/1000
    electricity_price = float(electricity_price)

    # Pass the values to your function
    multi_simulation.run(agrilculture_tax_rate, logistic_tax_rate, manufacturing_tax_rate, tech_tax_rate, electricity_price)

    return jsonify(message='Simulation run successfully'), 200


if __name__ == '__main__':
    app.run(debug=True)