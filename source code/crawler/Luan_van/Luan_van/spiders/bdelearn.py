"""from scrapy.spiders import CrawlSpider, Rule
from Luan_van.items import BangDiemItem
from scrapy.http    import Request
import scrapy
from scrapy.linkextractors import LinkExtractor


class ESpider(CrawlSpider):
    name = "bdelearn"
    allowed_domains = ['elearning-cse.hcmut.edu.vn']
    start_urls = ['https://elearning-cse.hcmut.edu.vn/portal/tool/891f0b75-bf13-4590-b2b2-0a6df7de3544?assignmentId=/assignment/a/61a3aabd-cdf6-4715-ab6c-b3353021559c/e6e96d7c-1044-456f-8317-bb8b7594fec0&panel=Main&sakai_action=doGrade_assignment']

    '''def start_requests(self):
        for url in self.start_urls:
            yield SplashRequest(url, self.parse)
            '''
    def parse(self, response):
        return scrapy.FormRequest.from_response(
            response,
            formdata={'eid': '1412961', 'pw': 'GiaPhuc1808'},
            callback=self.parse_page
            )
    

    def temp(self, response):
        yield Request(response.url, self.parse_page)
    def parse_page(self, response):
        '''scrapy.FormRequest.from_response(
            response,
            formxpath='//select[contains(@id, "gbForm:pager_pageSize")]',
            formname = "gbForm",
            formdata={'gbForm:pager_pageSize': "0"})'''
        print("dasfasfa")
        print("RESPONSE", response.xpath('//body').extract())
        #names = response.xpath('//div[contains(@id, "contents")]/div[contains(@id, "q3")]/div//tbody/tr/td/a/text()').extract()
        names = response.xpath('//td[contains(@header, "studentname")]').extract()
        #scores = response.xpath('//span[contains(@class, "courseGrade")]/text()').extract()
        date = response.xpath('//td[contains(@header, "sumitted")]').extract()
        for item in zip(names, date):
            bditem = BangDiemItem()
            bditem['a_name'] = item[0]
            bditem['b_date'] = item[1]
            yield bditem"""
