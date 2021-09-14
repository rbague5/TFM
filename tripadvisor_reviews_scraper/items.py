# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from scrapy.item import Item, Field


class TripadvisorReviewsScraperItem(Item):

    user_id = Field()
    review_id = Field()
    title_review = Field()
    date = Field()
    parse_count = Field()
    author = Field()
    review_preview = Field()
    review_full = Field()
    rating_review = Field()
    sample = Field()
    restaurant_name = Field()
    city = Field()
    url_restaurant = Field()
    url_review = Field()
    pass


class TripadvisorRestaurantCharacteristicItem(Item):
    restaurant_id = Field()
    restaurant_url = Field()
    restaurant_name = Field()
    restaurant_ubication = Field()
    cuisines = Field() # tipos de comida
    meals = Field() #comidas
    features = Field() # ventajas
    price_range = Field() # rango de precios
    special_diets = Field() # dietas especiales
    restaurant_details = Field()
    pass