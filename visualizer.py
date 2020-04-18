"""
This class takes in individuals and writes them to a .csv for draw.io to process

Use:

$python3.6 visualize.py gen11_pop.npy

Visualizer will take problem.py's SEED_ROOT_DIR constant and find the generation and output a CSV file in the same directory

Open draw.io
Go to Arrange > Insert > Advanced > CSV...
Then just copy-paste the csv file content into the textbox

Note: Fitnesses here are the accuracy values and F-1 score.
"""

import string
import numpy as np

import os
import sys

header = '## Hello World \
            \n# label: %title%<br><i style="color:gray;">%text%</i> \
            \n# style: html=1;shape=rectangle;rounded=1;fillColor=%fill%;strokeColor=%fill% \
            \n# namespace: csvimport- \
            \n# connect: {"from":"refs", "to":"id", "invert":true, "style":"curved=0;endArrow=blockThin;endFill=1;"} \
            \n# width: auto \
            \n# height: auto \
            \n# padding: 10 \
            \n# ignore: id,fill,refs \
            \n# nodespacing: 10 \
            \n# levelspacing: 30 \
            \n# edgespacing: 40 \
            \n# layout: horizontalflow \
            \n## CSV starts under this line \
            \nid,title,text,fill,refs \
            \n '


class Visualizer:
    def __init__(self, output_path='vis'):
        # Limit on number of block is letters of alphabet
        self.output_path = output_path
        self.shifts = list(string.ascii_lowercase)
        self.colors = ['#dae8fc', '#f8cecc', "#d5e8d4"] * 9
        self.header = header
        self.arrow_color = "#1500ff"
        self.csv_rows = self.header.split("\n")
        self.individual_num = 0
        self.append_csv('w+')

    def add_to_csv(self, individual, from_npy=False):
        self.csv_rows = []
        self.individual_num += 1
        prev_output = ""
        #num_blocks = len(individual.blocks) #if not from_npy else len(individual[:-2])
        count = 0
        for block in individual.blocks:

            active_nodes = block.active_nodes
            # else:
            #     active_nodes = {-1, block[-2], len(block) - 2}
            #     open = [block[-2]]
            #     while len(open) != 0:
            #         s = open.pop(0)
            #         if s != -1:
            #             print(block[s])
            #             open += block[s]['inputs']
            #             active_nodes.update(block[s]['inputs'])
            #     active_nodes = sorted(list(active_nodes))

            for active_node in active_nodes:
                self.format_node(block, count, active_node, prev_output)
            prev_output = f"{self.individual_num}{self.shifts[count]}{active_nodes[-1]}"
            count += 1

        loss, f1 = individual.fitness.values
        self.csv_rows.append(f"{self.individual_num}END,\"Fitness: ({1 - loss},{f1})\",,{'#ffe6cc'},\"{prev_output}\"")

        self.append_csv()

    def format_node(self, block, block_num, node_index, prev_output=""):
        block_id = f"{self.individual_num}{self.shifts[block_num]}"
        node = {
            'id': f"{block_id}{node_index}",
            'title': "",
            'text': "",
            'fill': self.colors[block_num],
            'refs': "",
        }

        fn = block[node_index]
        if type(fn) == str:  # Input
            node.update({'title': fn, 'refs': prev_output})
        elif type(fn) == np.int64:  # Output
            node.update({'title': "Output", 'refs': f"{block_id}{fn}"})
        else:  # Normal
            node.update({'title': fn['ftn'].__name__, 'refs': ','.join([block_id + str(x) for x in fn['inputs']])})

        self.csv_rows.append(f"{node['id']},{node['title']},{node['text']},{node['fill']},\"{node['refs']}\"")

    def append_csv(self, mode='a'):
        with open(self.output_path + ".csv", mode, newline='') as csv:
            for row in self.csv_rows:
                csv.write(row + '\n')
        print("append")


if __name__ == "__main__":
    # load file and path
    #filename, file_extension = os.path.splitext(sys.argv[1])
    filename = 'vis_out'
    path = 'outputs/problem_symbolicRegression/20200418-143513/univ0' + '/' + filename
    #print(path)
    print('Saving visualized csv to {}'.format(path + '.csv'))

    # create visualizer object and load population
    vis = Visualizer(path)
    population = np.load('outputs/problem_symbolicRegression/20200418-143513/univ0/gen10_pop.npy', allow_pickle=True)
    print(len(population))
    for individual in population:
        vis.add_to_csv(individual, from_npy=True)

    with open(path + ".csv", 'r', newline='') as csv:
        for row in csv:
            print(row)

    # we can adjust the code above to visualize the best individual trained
    # with many epochs from evaluator.py
    # it'd be the same shape as one of the individuals above, but with different fitness values
    #
    # idea is the same as manually copy pasting into output csv file's individual's fitness
