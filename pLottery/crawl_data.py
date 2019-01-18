# -*- coding: utf-8 -*-

import urllib
import urllib2
from bs4 import BeautifulSoup

urlDict = {
    'dlt':'http://kaijiang.500.com/static/info/kaijiang/xml/dlt/list.xml',
    'ssq': 'http://kaijiang.500.com/static/info/kaijiang/xml/ssq/list.xml'
}
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/66.0.3359.181 Chrome/66.0.3359.181 Safari/537.36"
dataDir = './data/'


def save_info(raw, l_type):
    soup = BeautifulSoup(raw, 'html.parser', from_encoding='utf-8')
    nodes = soup.select('row')
    file_name = dataDir+l_type
    file_str = ''
    with open(file_name, 'w') as fp:
        for node in nodes:
            period = node['expect']
            open_codes = node['opencode']
            open_codes = open_codes.replace('|', ',')
            open_time = node['opentime'] if node['opentime'] else ''
            open_time = open_time.split(' ')[0]
            file_str += (period+'('+open_time+'):'+open_codes+'\n')
        fp.write(file_str)


def crawl_info(l_type):
    req = urllib2.Request(urlDict[l_type], headers={'User-Agent': UA})
    resp = urllib2.urlopen(req)
    raw = resp.read()
    save_info(raw, l_type)


def get_lottery_info(l_type):
    l_type = l_type if l_type else 'ssq'
    crawl_info(l_type)


if __name__ == '__main__':
    get_lottery_info('ssq')
