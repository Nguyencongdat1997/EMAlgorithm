import requests
import json


class APIConnector(object):
    def __init__(self):
        self.url = 'https://24zl01u3ff.execute-api.us-west-1.amazonaws.com/beta'

    def get_json_attribute(self, data, attribute, default_value):
        return data.get(attribute) or default_value

    def get_one_draw(self):
        response = requests.get(url=self.url)

        flips = json.loads(self.get_json_attribute(response.json(), 'body', '[-1]').encode('utf-8'))
        assert len(flips) == 20, 'Invalid api response.'
        return flips
