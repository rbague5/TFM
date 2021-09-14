import time
import urllib

import scrapy, csv
import pandas as pd
from scrapy import Selector
from scrapy.exceptions import CloseSpider
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException, \
    StaleElementReferenceException
from selenium.webdriver.chrome.options import Options
from urllib3.exceptions import NewConnectionError
from scrapy.selector import Selector
from scrapy.utils.response import open_in_browser

from ..items import TripadvisorRestaurantCharacteristicItem

# Variables globales
DELAY_ALLOWED = 0.5  # en segundos, puede ser necesario multiplicarlo por un factor, según velocidad de recuperación
CITY = "gijon"

def read_csv_url(f):
    df = pd.read_csv(f)
    return df['restaurantUrl'].tolist()


class TripadvisorRestaurantDetailsSpider(scrapy.Spider):

    file = './urls/restaurantUrl_' + CITY + '.csv'
    urls = read_csv_url(f=file)
    # print(urls)
    # time.sleep(5)
    name = 'tripadvisorRestaurantDetails'
    allowed_domains = ['tripadvisor.com']

    start_urls = urls
    # start_urls = ['https://www.tripadvisor.com/Restaurant_Review-g187451-d4829043-Reviews-Sidreria_Casa_Fede-Gijon_Asturias.html']

    avoids = ('.*\/(robots.txt)$')

    # Controlar el numero de requests en paralelo
    # https://doc.scrapy.org/en/latest/topics/settings.html#concurrent-requests

    custom_settings = {
        # specifies exported fields and order
        'FEED_EXPORT_FIELDS': ["restaurant_id", "restaurant_url", "restaurant_name", "restaurant_ubication",
                               "price_range", "special_diets", "meals", "cuisines", "features", "restaurant_details"],
    }

    # Comprobar los regex
    # https://regex101.com/
    # Recordar xpath
    # https://docs.scrapy.org/en/xpath-tutorial/topics/xpath-tutorial.html
    def __init__(self, num_reviews=100, *a, **kw):

        super(TripadvisorRestaurantDetailsSpider, self).__init__(*a, **kw)
        self.crawled_reviews = 0
        self.max_reviews = num_reviews

        options = Options()
        options.headless = True
        options.add_argument("--lang=en-US")

        # Preferences:
        # (1) idiomas permitidos
        # (2) Evitar cargar imagenes por default, speed-up
        # (3) Evitar cargar imagenes, otra opcion

        # preferences = {'intl.accept_languages': 'en,en_US',
        #                "profile.default_content_settings.images": 2,
        #                "profile.managed_default_content_settings.images": 2}
        preferences = {'intl.accept_languages': 'en,en_US',
                        "profile.default_content_settings.images": 2,
                        "profile.managed_default_content_settings.images": 2}

        options.add_experimental_option('prefs', preferences)

        # Chrome driver
        self.driver = webdriver.Chrome(chrome_options=options)

        # Check cookies banner
        self.cookie_banner = True

    def parse(self, response):
        self.logger.info('A response from %s just arrived!', response.url)
        # open_in_browser(response)

        # https://stackoverflow.com/questions/45384382/scrapy-select-xpath-with-a-regular-expression
        # https://stackoverflow.com/questions/614797/xpath-find-a-node-that-has-a-given-attribute-whose-value-contains-a-string
        # https://stackoverflow.com/questions/37732649/scrapy-loop-xpath-selector-escaping-object-it-is-applied-to-and-returning-all

        # --------------------------------------------------------------------------------------------------
        try:
            self.driver.get(response.url + "?hl=en")
            print(" == START INFO == ")
            print("PÁGINA RECUPERADA EXITOSAMENTE")
            print(str(response.url) + "?hl=en")
            print(" == END INFO == ")

        except NewConnectionError:
            print(" == START ERROR == ")
            print("Fallo al realizar el intento de conexión a: " + str(response.url))
            print(" == END ERROR == ")

        try:
            if self.cookie_banner:
                try:
                    # Checks if Cookies settings obscures page!
                    self.driver.find_element_by_xpath(
                        '//button[contains(@id, "_evidon-accept-button")]'
                    ).click()
                    print("== START INFO ==")
                    print(" == COOKIES BANNER IS CLICKED TO CLOSE DIALOG!")
                    print("== END INFO ==")
                    self.cookie_banner = False
                except NoSuchElementException:
                    self.cookie_banner = False

            time.sleep(DELAY_ALLOWED * 2)

        except ElementClickInterceptedException:
            print("== START ERROR ==")
            print(" == COOKIES BANNER IS ANNOYING POOR SCRAPPER :(")
            print("== END ERROR ==")
            self.cookie_banner = True

        try:
            self.driver.find_element_by_xpath(
                '//div[@class="y5QNMrR5"]//a[@class="_3xJIW2mF"]').click()

            print(" == START INFO == ")
            print(" == More buttom for more details about the restaurant CLICKED == ")
            print(" == END INFO == ")
            time.sleep(DELAY_ALLOWED * 2)

        except NoSuchElementException:
            print("== START ERROR ==")
            print("Ningún desplegable disponible en la página de comentarios: " + str(response.url))
            print("== END ERROR ==")

        # --------------------------------------------------------------------------------------------------
        sel = Selector(text=self.driver.page_source)
        # --------------------------------------------------------------------------------------------------
        restaurant_details = sel.xpath('//div[@class="_1RdgGpnh"]//text()').getall()

        # # guardamos los detalles de puntiaciones, detalles ubicacion y contactos
        # full_details = response.xpath(
        #         '//div[contains(@data-tab, "TABS_OVERVIEW")]//div[contains(@class, "ui_columns")]')
        #
        # restaurant_details = full_details.xpath('//div[@class="_3acGlZjD"]//div[@class="_2NeF0VlQ"]//div[@class="_3UjHBXYa"]//div//div//text()').getall()
        #
        print(f"Len: {len(restaurant_details)} --> {restaurant_details}")

        # se compueba si pilla los items, si no los pilla es q ha redirigido a la pagina de la ciudad
        if len(restaurant_details) > 0:

            # variables for detaiils
            rango_precios = "None"
            tipos_cocina = "None"
            comidas = "None"
            ventajas = "None"
            dietas_especiales = "None"

            for detail in restaurant_details:
                if detail == "PRICE RANGE":
                    index = restaurant_details.index("PRICE RANGE")

                    rango_precios = restaurant_details[index+1]
                    rango_precios.replace(u'\xa0', u' ')
                elif detail == "CUISINES":
                    index = restaurant_details.index("CUISINES")
                    tipos_cocina = restaurant_details[index+1]
                elif detail == "Meals":
                    index = restaurant_details.index("Meals")
                    comidas = restaurant_details[index+1]
                elif detail == "FEATURES":
                    index = restaurant_details.index("FEATURES")
                    ventajas = restaurant_details[index+1]
                elif detail == "Special Diets":
                    index = restaurant_details.index("Special Diets")
                    dietas_especiales = restaurant_details[index + 1]

            print(f"rango_precios: {rango_precios}")
            print(f"tipos_cocina: {tipos_cocina}")
            print(f"comidas: {comidas}")
            print(f"ventajas: {ventajas}")
            print(f"dietas_especiales: {dietas_especiales}")
            print(f"url : {response.url}")

            url = response.url

            splitted_url = str(url).split("-")  # separamos las url de los restaurantes por el guion
            restaurant_id = splitted_url[2]
            restaurant_name = splitted_url[4]
            restaurant_ubication = splitted_url[5].split(".")[0]

            print(f"restaurant_id : {restaurant_id}")
            print(f"restaurant_name : {restaurant_name}")
            print(f"restaurant_ubication : {restaurant_ubication}")


            review_item = TripadvisorRestaurantCharacteristicItem()

            review_item['restaurant_id'] = restaurant_id
            review_item['restaurant_url'] = url
            review_item['restaurant_name'] = restaurant_name
            review_item['restaurant_ubication'] = restaurant_ubication
            review_item['cuisines'] = tipos_cocina
            review_item['meals'] = comidas
            review_item['features'] = ventajas
            review_item['price_range'] = rango_precios
            review_item['special_diets'] = dietas_especiales
            review_item['restaurant_details'] = restaurant_details

            yield review_item

        else:
            # aquí entra si redicire a la pagina de la cuidad o si el restaurante no contiene detalles
            print(f"Len: {len(restaurant_details)} --> {restaurant_details}")
            url = str(response.url)
            if 'Review' in url:
                # cuando en la url hau 'Review' significa q si que el restaurante existe, pero este no tiene detalles
                f = open('./details/restaurant_without_details_' + CITY + '_.txt', 'a', encoding='UTF8')
                # create the csv writer
                f.write(url + "\n")
                f.close()

                url = response.url

                splitted_url = str(url).split("-")  # separamos las url de los restaurantes por el guion
                restaurant_id = splitted_url[2]
                restaurant_name = splitted_url[4]
                restaurant_ubication = splitted_url[5].split(".")[0]

                # variables for detaiils
                rango_precios = "None"
                tipos_cocina = "None"
                comidas = "None"
                ventajas = "None"
                dietas_especiales = "None"
                restaurant_details = "None"

                review_item = TripadvisorRestaurantCharacteristicItem()

                review_item['restaurant_id'] = restaurant_id
                review_item['restaurant_url'] = url
                review_item['restaurant_name'] = restaurant_name
                review_item['restaurant_ubication'] = restaurant_ubication
                review_item['cuisines'] = tipos_cocina
                review_item['meals'] = comidas
                review_item['features'] = ventajas
                review_item['price_range'] = rango_precios
                review_item['special_diets'] = dietas_especiales
                review_item['restaurant_details'] = restaurant_details

                yield review_item

        # scrapy crawl tripadvisorRestaurantDetails -o ./details/london_details_full.csv -t csv


    def no_parse(self, response):
        print(" == START INFO == ")
        print("No parse: " + str(response.url))
        print(" == END INFO == ")