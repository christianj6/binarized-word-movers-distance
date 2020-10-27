import requests
from bwmd.compressor import load_vectors
import bs4
import dill
import re


def preprocess_wikipedia(wordlist):
    '''
    Preprocess and clean wikipedia pages
    that they can be employed for
    evaluation on the triplets task.

    Parameters
    ---------
        wordlist : list[str]
            List of words that we only extract
            those we expect to find in the
            vectors.
    '''
    def process_content(url):
        '''
        Parse a single url, extracting the
        relevant content.

        Parameters
        ---------
            url : str
                Path to webpage.

        Returns
        ---------
            cleaned_txt : str
                Extracted text with irrelevant
                characters and words removed.
        '''
        # Get response.
        response = requests.get(url)
        # Get content.
        contents = response.content
        if response is not None:
            # Get response as parsed html text.
            html = bs4.BeautifulSoup(response.text, 'html.parser')
            # Extract the title.
            title = html.select("#firstHeading")[0].text
            # Get all the paragraphs.
            paragraphs = html.select("p")

        # Remove non alphanumeric characters.
        paragraphs = [p.text.strip() for p in paragraphs]
        paragraphs = [re.sub(r'[^a-zA-Z0-9_ ]+', '', p) for p in paragraphs]
        # Remove words not in wordlist.
        clean_for_wordlist = lambda x: ' '.join([x for x in x.split() if x in wordlist])
        paragraphs = list(map(clean_for_wordlist, paragraphs))

        return ''.join([title, ' ']+paragraphs)

    with open('res\\datasets\\triplets\\wikipedia.txt', 'r') as f:
        # Just use a counter to name the files.
        counter = 0
        for line in f:
            print(counter)
            try:
                # Get processed text of each article.
                abc = map(process_content, tuple(line.split()))
                # Dump the tuple to file.
                with open(f'res\\datasets\\triplets\\wikipedia-{counter}', 'wb') as g:
                    dill.dump(tuple(abc), g)

                counter += 1

            except Exception:
                continue


def main():
    # Load vectors so we can exclude words not in vector space.
    vectors, words = load_vectors(
            "res\\glove-256.txtc",
            expected_dimensions=256,
            expected_dtype='bool_',
            get_words=True
        )
    # Process and save wikipedia articles.
    preprocess_wikipedia(words)


if __name__ == '__main__':
    main()

