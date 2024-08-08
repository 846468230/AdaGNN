class SearchSpace(object):
    """
    Loading the search space dict
    """
    def __init__(self, args=None):
        self.gnn_layers = int(args.gnn_layers)
        self.stack_gnn_architecture = ['attention-aggregation-hidden_dimension', 'activation'] * self.gnn_layers
        self.space_dict = {
            'attention-aggregation-hidden_dimension': [
                                                       'gcn-max-64', 'gcn-mean-64', 'gcn-add-64',
                                                       'gat-max-64', 'gat-mean-64', 'gat-add-64',
                                                       'graphsage-max-64', 'graphsage-mean-64', 'graphsage-add-64',
                                                       'gin-max-64', 'gin-mean-64', 'gin-add-64',
                                                       'rgcn-max-64', 'rgcn-mean-64', 'rgcn-add-64',
                                                       'rgt-max-64', 'rgt-mean-64', 'rgt-add-64',
                                                       'hgt-max-64', 'hgt-mean-64', 'hgt-add-64',
                                                       'simplehgn-max-64', 'simplehgn-mean-64', 'simplehgn-add-64',

                                                       'gcn-max-128', 'gcn-mean-128', 'gcn-add-128',
                                                       'gat-max-128', 'gat-mean-128', 'gat-add-128',
                                                       'graphsage-max-128', 'graphsage-mean-128', 'graphsage-add-128',
                                                       'gin-max-128', 'gin-mean-128', 'gin-add-128',
                                                       'rgcn-max-128', 'rgcn-mean-128', 'rgcn-add-128',
                                                       'rgt-max-128', 'rgt-mean-128', 'rgt-add-128',
                                                       'hgt-max-128', 'hgt-mean-128', 'hgt-add-128',
                                                       'simplehgn-max-128', 'simplehgn-mean-128', 'simplehgn-add-128',
            ],

                                                       # 'gcn-max-256', 'gcn-mean-256', 'gcn-add-256',
                                                       # 'gat-max-256', 'gat-mean-256', 'gat-add-256',
                                                       # 'graphsage-max-256', 'graphsage-mean-256', 'graphsage-add-256',
                                                       # 'gin-max-256', 'gin-mean-256', 'gin-add-256',
                                                       # 'rgcn-max-256', 'rgcn-mean-256', 'rgcn-add-256',
                                                       # 'rgt-max-256', 'rgt-mean-256', 'rgt-add-256',
                                                       # 'hgt-max-256', 'hgt-mean-256', 'hgt-add-256',
                                                       # 'simplehgn-max-256', 'simplehgn-mean-256', 'simplehgn-add-256'],
            # 'attention-aggregation-hidden_dimension': ['gat-max-32',
            #                                            'rgcn-mean-256'],

            'activation': ['elu', 'leaky_relu', 'linear', 'relu', 'relu6', 'sigmoid', 'softplus', 'tanh'],
        }
        self.update_space_dict(args)
        print(self.space_dict)

    def update_space_dict(self, args):
        if args.attention is not None:
            self.space_dict['attention-aggregation-hidden_dimension'] = [_ for _ in self.space_dict['attention-aggregation-hidden_dimension'] if _.split('-')[0] == args.attention]
        if args.activation is not None:
            self.space_dict['activation'] = [_ for _ in self.space_dict['activation'] if _ == args.activation]
        if args.hidden_dimension is not None:
            self.space_dict['attention-aggregation-hidden_dimension'] = [_ for _ in self.space_dict['attention-aggregation-hidden_dimension'] if _.split('-')[2] == args.hidden_dimension]
        if args.aggregation is not None:
            self.space_dict['attention-aggregation-hidden_dimension'] = [_ for _ in self.space_dict['attention-aggregation-hidden_dimension'] if _.split('-')[1] == args.aggregation]
