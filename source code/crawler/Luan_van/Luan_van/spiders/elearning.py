from scrapy.spiders import CrawlSpider, Rule
from Luan_van.items import LuanVanItem
from scrapy.http    import Request
import scrapy
from scrapy.linkextractors import LinkExtractor

class MySpider(CrawlSpider):
    name = "elearning"
    #start_urls = ["https://elearning-cse.hcmut.edu.vn/portal/login"]
    start_urls = ["https://elearning-cse.hcmut.edu.vn/portal/tool/45060f5f-4cf2-4d69-8bef-80edbf114c3b/forums/list.page"]
    allowed_domains = ['elearning-cse.hcmut.edu.vn']
    #rules = (Rule(LinkExtractor(allow=(), restrict_css=('a.forumlink')),callback='parse', follow=False),)

    def parse(self, response):
        return scrapy.FormRequest.from_response(
            response,
            formdata={'eid': '1412961', 'pw': 'GiaPhuc1808'},
            callback=self.after_login
            )

    def after_login(self, response):
        self.logger.error("Login succeeded!")
        #baseUrl = 'https://elearning-cse.hcmut.edu.vn/portal/tool/c4e39edb-1580-405b-a88a-f5b5e368523c/posts/list/5824.page'
        #yield Request(url= baseUrl, callback=self.parse_page)
        links = LinkExtractor(allow=("/portal/tool/45060f5f-4cf2-4d69-8bef-80edbf114c3b/(posts|forums)/"), restrict_css=('a.forumlink')).extract_links(response)
        #links = LinkExtractor(allow=("/portal/tool/c4e39edb-1580-405b-a88a-f5b5e368523c/(posts|forums)/.*\.page$")).extract_links(response)
        print(links)
        print(len(links))
        for url in links:
            yield Request(url=url.url, callback=self.parse_page)

    
    
    def parse_page(self, response):
        links = LinkExtractor(allow=("/portal/tool/45060f5f-4cf2-4d69-8bef-80edbf114c3b/(posts|forums)/.*\.page$")).extract_links(response)
        crawledLink = []
        for link in links:
            url = link.url
            if url not in crawledLink:
                crawledLink.append(url)
                yield Request(url, self.parse_page)
        names = response.xpath('//td/span[contains(@class, "name")]/b/text()').extract()
        names = list(map(lambda x: x.replace(u"\xa0", ""), names))
        dates = response.xpath('//td[contains(@class, "postInfo")]/span[contains(@class, "postdetails")]/text()').extract()
        dates = list(filter(lambda x: len(x) > 25, dates))
        topics = response.xpath('//td[contains(@class, "postInfo")]/span[contains(@class, "postdetails")]/a/text()').extract()
        topics = list(filter(lambda x: '\n' not in x, topics))
        #contents = response.xpath('//td[contains(@class, "rowUnread")]/span[contains(@class, "postbody")]/text()').extract()
        #contents = response.css('td.rowUnread > span.postbody::text').extract_first()
        contents = []
        for post in response.css('td > span.postbody'):
            content = post.css('::text').extract()
            content = list(map(lambda x: x.strip(), content))
            content = list(filter(lambda x: len(x) != 0, content))
            content = ' '.join(content).replace(u"\xa0", "")
            contents.append(content)
        for item in zip(names, dates, topics, contents):
            lvitem = LuanVanItem()
            name = item[0]
            if(name[len(name) - 1] == " "):
                print("NAME", name)
                name = name[:len(name)-1]
            lvitem["a_name"] = name
            lvitem["b_date"] = item[1][0:10]
            lvitem["c_topic"] = item[2]
            lvitem["d_content"] = item[3]
            return lvitem
        
