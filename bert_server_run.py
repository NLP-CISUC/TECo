
from bert_serving.server import BertServer, get_args_parser

if __name__ == '__main__':
    args = get_args_parser().parse_args(['-model_dir', 'models_db/bert_pretrained_models/multi_cased_L-12_H-768_A-12',
                                         '-port', '5555',
                                         '-port_out', '5556',
                                         '-max_seq_len', 'NONE',
                                         '-mask_cls_sep',
                                         '-pooling_strategy', 'NONE',
                                         '-show_tokens_to_client',
                                         '-cpu'])
    server = BertServer(args)
    server.start()
    """
    active = True
    while active:
        variable = input('Please enter a value: ')
        if variable is not None:
            server.shutdown(args=args)
            break"""
