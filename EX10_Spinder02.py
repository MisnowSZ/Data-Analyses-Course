# coding:utf-8
import json
import requests as req
from lxml import etree
from selenium import webdriver
import os

req_url = 'https://search.douban.com/movie/subject_search?search_text=宫崎骏&cat=1002'
src_xpath = "//div[@class='item-root']/a[@class='cover-link']/img[@class='cover']/@src"
title_xpath = "//div[@class='item-root']/div[@class='detail']/div[@class='title']/a[@class='title-text']"
driver = webdriver.Chrome("D:\\Python\\chromedriver")
driver.get(req_url)
html = etree.HTML(driver.page_source)
srcs = html.xpath(src_xpath)
print(srcs)
picpath = "D:\\Python\\Learning\\spider"
def download(src, id):
	id = id.replace(u'\u200e', u'').replace(u'?', u'').replace(u'/', u'or')
	dic = picpath + '\\' + str(id) + '.webp'
	try:
		pic = req.get(src, timeout = 30)
		fp = open(dic, 'wb')
		fp.write(pic.content)
		fp.close()
	except req.exceptions.ConnectionError:
		print("Can not download pict")

for i in range(0, 150, 15):
	url = req_url + '&start=' + str(i)
	driver.get(url)
	html = etree.HTML(driver.page_source)
	srcs = html.xpath(src_xpath)
	titles = html.xpath(title_xpath)
	for src, title in zip(srcs, titles):
		download(src, title.text)
