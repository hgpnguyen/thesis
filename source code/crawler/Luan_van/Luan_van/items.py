# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class LuanVanItem(scrapy.Item):
    # define the fields for your item here like:
    a_name = scrapy.Field()
    b_date = scrapy.Field()
    c_topic = scrapy.Field()
    d_content = scrapy.Field()

class BangDiemItem(scrapy.Item):
    a_name = scrapy.Field()
    b_diem = scrapy.Field()

class Asiignment(scrapy.Item):
    a_name = scrapy.Field()
    b_date = scrapy.Field()
