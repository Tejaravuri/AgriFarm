def generate_care_advice(expected_yield, rainfall, temperature, soil_ph=6.5):
    advice = []

    if rainfall < 800:
        advice.append("Increase irrigation frequency, especially during flowering stage.")

    if temperature > 35:
        advice.append("Use mulching to reduce soil moisture loss.")

    if soil_ph < 6.0:
        advice.append("Apply lime to improve soil pH.")

    if expected_yield < 2.5:
        advice.append("Use organic manure and practice crop rotation to improve soil fertility.")

    if not advice:
        advice.append("Current conditions are optimal. Continue regular monitoring.")

    return advice
