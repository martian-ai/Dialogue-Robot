import time

# import url_table as url_table_lib
from tools.url_table import get_url_table
# import urllib2
# import urllib
import urllib.request
from tools.logger import data_analysis_logger as logger


def crawler(cid):
    """
    Args:
        cid: int, crawler id
    """
    url_table = get_url_table()
    while True:
        start_t = time.time()
        url_node = url_table.get()
        addinfourl = urllib.request.urlopen(url_node.url)
        url_table.task_done()
        end_t = time.time()
        logger.info("Crawler #%s crawl %s done, cost %s ms" % (cid, 
                     url_node.url,
                     int(end_t*1000 - start_t*1000) ))

