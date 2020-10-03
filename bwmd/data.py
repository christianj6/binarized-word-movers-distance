import requests
import bs4


# TODO: Load vectors so we can exclude words not in vector space.

def preprocess_wikipedia():
    '''
    '''
    def process_content(url):
        '''
        '''
        response = requests.get(url)
        contents = response.content

        if response is not None:
            html = bs4.BeautifulSoup(response.text, 'html.parser')
            title = html.select("#firstHeading")[0].text
            paragraphs = html.select("p")

        # TODO: Clean for [digits]

        return ''.join([title, ' ']+[p.text.strip() for p in paragraphs])

    with open('res\\datasets\\triplets\\wikipedia.txt', 'r') as f:
        for line in f:
            a, b, c = tuple(line.split())
            print(process_content(a))
            print('\n')
            print(process_content(b))
            print('\n')
            print(process_content(c))
            print('\n')

            # TODO: Save the documents in appropriate format, which
            # represents the correct semantic relationships between them and
            # can be easily retrieved during testing.

if __name__ == '__main__':
    # TODO: arxiv is totally unsuitable
    # TODO: Process wikipedia
    preprocess_wikipedia()
    # TODO: Save everything in the res.
