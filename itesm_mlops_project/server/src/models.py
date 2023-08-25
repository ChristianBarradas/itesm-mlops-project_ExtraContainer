from pydantic import BaseModel


class HousePricePrediction (BaseModel):
    """
    Representa los atributos para poder realizar el pronostico del precio de la casa
    Attributes:
        bedrooms
        bathrooms
        sqft_living
        sqft_lot
        floors
        waterfront
        view
        condition
        grade
        sqft_above
        sqft_basement
        yr_built
        yr_renovated
        lat
        long
        sqft_living15
        sqft_lot15
        month
        year
    """

    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: float
    view: float
    condition: float
    grade: float
    sqft_above: float
    sqft_basement: float
    yr_built: float
    yr_renovated: float
    lat: float
    long: float
    sqft_living15: float
    sqft_lot15: float
    month: float
    year: float
