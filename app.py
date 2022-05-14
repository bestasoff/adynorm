import argparse
import json
import logging
import tornado.web
import tornado.ioloop
import tornado.autoreload

from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification

from adynorm.adynorm import Adynorm, AdynormNet
from adynorm.datasets import (
    DictDataset
)
from evaluator import Evaluator

logging.basicConfig(
    filename='.server.log',
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

parser = argparse.ArgumentParser(description='Adynorm Demo')

# Required
parser.add_argument('--model_name_or_path', required=True, help='Directory for NER model')
parser.add_argument('--ner_model_path', required=True, help='Directory for NER model')
parser.add_argument('--entity_classifier_path', required=True, help='Directory for EC model')
parser.add_argument('--adynorm_path', required=True, help='Directory for adynorm model')
parser.add_argument('--adynorm_net_path', required=True, help='Directory for adynorm_net model')

# Settings
parser.add_argument('--port', type=int, default=8888, help='port number')
parser.add_argument('--max_length', type=int, default=25)
parser.add_argument('--show_predictions', action="store_true")
parser.add_argument('--val_dict_path', type=str, default=None, help='dictionary path')

args = parser.parse_args()

device = 'cpu'

labels = ['B', 'I', 'O']
labels = [l + "-bio" if l != 'O' else l for l in labels]
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}
num_labels = len(labels)

config = AutoConfig.from_pretrained(
    args.model_name_or_path,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path
)

ner_model = AutoModelForTokenClassification.from_pretrained(
    args.ner_model_path,
    config=config
).to(device)

entity_classifier = AutoModelForSequenceClassification.from_pretrained(
    args.model_name_or_path,
).to(device)

adynorm = Adynorm(
    max_length=args.max_length,
    device=device
)
adynorm.load(args.adynorm_path, args.adynorm_path)
adynorm.encoder.to(device)
adynorm_net = AdynormNet(encoder=adynorm.get_encoder()).to(device)

ner_model.eval()
entity_classifier.eval()
adynorm_net.eval()

dictionary = DictDataset(args.val_dict_path).data

evaluator = Evaluator(
    ner_model, entity_classifier, adynorm, adynorm_net,
    tokenizer, id2label, label2id, dictionary, 'datasets/dictionary_embeddings.pt'
)


def recognize_and_normalize(sentence):
    predictions = evaluator(sentence)
    # logging.info('predictions!{}'.format({
    #     predictions,
    # }))
    return predictions


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("./template/index.html")


class NormalizeHandler(tornado.web.RequestHandler):
    def get(self):
        string = self.get_argument('string', '')
        logging.info('get!{}'.format({
            'string': string,
        }))
        self.set_header("Content-Type", "application/json")
        output = recognize_and_normalize(sentence=string)
        print(output)
        self.write(json.dumps(output))


def make_app():
    settings = {
        'debug': True
    }
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/normalize/", NormalizeHandler),
        (r'/semantic/(.*)', tornado.web.StaticFileHandler, {'path': './semantic'}),
        (r'/images/(.*)', tornado.web.StaticFileHandler, {'path': './images'}),
    ], **settings)


if __name__ == '__main__':
    logging.info('Starting adynorm server at http://localhost:{}'.format(args.port))
    app = make_app()
    print('App made!')
    app.listen(args.port)
    tornado.ioloop.IOLoop.current().start()
