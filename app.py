from flask import Flask, render_template, request, jsonify
from model import get_prediction

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


# @app.route('/predict', methods=['POST', 'GET'])
# def predict():
#     # if requests.method == 'POST':
#     text = "Borrower and any endorsers or guarantors hereof severally waive presentment and demand for payment, notice of intent to accelerate maturity, protest or notice of protest and non-payment, bringing of suit and diligence in taking any action to collect any sums owing hereunder or in proceeding against any of the rights and properties securing payment hereunder, and expressly agree that this Note, or any payment hereunder, may be extended from time to time, and consent to the acceptance of further security or the release of any security for this Note, all without in any way affecting the liability of Borrower and any endorsers or guarantors hereof. No extension of time for the payment of this Note, or any installment thereof, made by agreement by Lender with any person now or hereafter liable for the payment of this Note, shall affect the original liability under this Note of the undersigned, even if the undersigned is not a party to such agreement."
#     result = get_prediction(text)
#     return result

@app.route('/get')
def get_bot_response():
    message = request.args.get('msg')
    response = ""
    if message:
        response = get_prediction(message)
        return str(response)
    else:
        return "Missing Data!"

if __name__ == '__main__':
    app.run()


"Borrower and any endorsers or guarantors hereof severally waive presentment and demand for payment, notice of intent to accelerate maturity, protest or notice of protest and non-payment, bringing of suit and diligence in taking any action to collect any sums owing hereunder or in proceeding against any of the rights and properties securing payment hereunder, and expressly agree that this Note, or any payment hereunder, may be extended from time to time, and consent to the acceptance of further security or the release of any security for this Note, all without in any way affecting the liability of Borrower and any endorsers or guarantors hereof. No extension of time for the payment of this Note, or any installment thereof, made by agreement by Lender with any person now or hereafter liable for the payment of this Note, shall affect the original liability under this Note of the undersigned, even if the undersigned is not a party to such agreement."
