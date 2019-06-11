import argparse
import os

from athene.retrieval.document.docment_retrieval import main as document_retrieval_main
#from athene.utils.config import Config
from common.util.log_helper import LogHelper


def document_retrieval(logger):
    # override args
    """
    logger.info("Starting document retrieval for training set...")
    document_retrieval_main(Config.db_path, Config.document_k_wiki, Config.raw_training_set, Config.training_doc_file,
                            Config.document_add_claim, Config.document_parallel)
    logger.info("Finished document retrieval for training set.")
    logger.info("Starting document retrieval for dev set...")
    document_retrieval_main(Config.db_path, Config.document_k_wiki, Config.raw_dev_set, Config.dev_doc_file,
                            Config.document_add_claim, Config.document_parallel)
    logger.info("Finished document retrieval for dev set.")
    logger.info("Starting document retrieval for test set...")
    document_retrieval_main(Config.db_path, Config.document_k_wiki, Config.raw_test_set, Config.test_doc_file,
                            Config.document_add_claim, Config.document_parallel)
    """
    document_retrieval_main(args.database, 7, args.infile, args.outfile, args.path_wiki_titles, True, True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile')
    parser.add_argument('--outfile')
    parser.add_argument('--database')
    parser.add_argument('--path_wiki_titles')
    args = parser.parse_args()
    
    LogHelper.setup()
    logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0])
    logger.info("=========================== Subtask 1. Document Retrieval ==========================================")
    print (args.database, args.infile, args.outfile)
    document_retrieval(logger)

