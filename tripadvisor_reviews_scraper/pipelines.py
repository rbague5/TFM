# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# Para el csv, indicarlo en otra ruta y mediante pipeline
# https://stackoverflow.com/questions/25163023/export-csv-file-from-scrapy-not-via-command-line

class TripadvisorReviewsScraperPipeline(object):
    def process_item(self, item, spider):
        return item
