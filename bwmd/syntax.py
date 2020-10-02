import networkx as nx
import matplotlib.pyplot as plt
import dill
import random
import math

def get_leacock_chodorow_distance(path, maxdepth):
    '''
    '''
    return -math.log(path / (2*maxdepth))


def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children)!=0:
            dx = width/len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap,
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


# Raw list of SpaCy tags.
tags = ['acl',
'acomp',
'advcl',
'advmod',
'agent',
'amod',
'appos',
'attr',
'aux',
'auxpass',
'case',
'cc',
'ccomp',
'compound',
'conj',
'cop',
'csubj',
'csubjpass',
'dative',
'dep',
'det',
'dobj',
'expl',
'intj',
'mark',
'meta',
'neg',
'nn',
'nounmod',
'npmod',
'nsubj',
'nsubjpass',
'nummod',
'oprd',
'obj',
'obl',
'parataxis',
'pcomp',
'pobj',
'poss',
'preconj',
'prep',
'prt',
'punct',
'quantmod',
'relcl',
'root',
'xcomp']

# TODO: Create network graph of SpaCy dependencies, which
# replicated the hierarchy present in the Stanford dependencies hierarchy.
G = nx.Graph()
# Add top-most nodes.
G.add_node("root")
G.add_node("dep")
G.add_node("meta")
G.add_edge("root", "dep")
G.add_edge("root", "meta")
# # Add aux node.
G.add_node("aux")
G.add_edge("dep", "aux")
G.add_node("auxpass")
G.add_edge("aux", "auxpass")
G.add_node("cop")
G.add_edge("aux", "cop")
# # Add arg node.
G.add_node("arg")
G.add_edge("dep", "arg")
G.add_node("agent")
G.add_edge("arg", "agent")
G.add_node("acomp")
G.add_edge("agent", "acomp")
G.add_node("pcomp")
G.add_edge("agent", "pcomp")
G.add_node("ccomp")
G.add_edge("agent", "ccomp")
G.add_node("xcomp")
G.add_edge("agent", "xcomp")
# Add objects.
G.add_node("obj")
G.add_edge("agent", "obj")
G.add_node("dobj")
G.add_edge("obj", "dobj")
G.add_node("attr")
G.add_edge("obj", "attr")
G.add_node("oprd")
G.add_edge("obj", "oprd")
G.add_node("obl")
G.add_edge("obj", "obl")
G.add_node("dative")
G.add_edge("obj", "dative")
G.add_node("pobj")
G.add_edge("obj", "pobj")
# Add subjects.
G.add_node("subj")
G.add_edge("agent", "subj")
G.add_node("nsubj")
G.add_edge("subj", "nsubj")
G.add_node("nsubjpass")
G.add_edge("nsubj", "nsubjpass")
G.add_node("csubj")
G.add_edge("subj", "csubj")
G.add_node("csubjpass")
G.add_edge("csubj", "csubjpass")
# Add some intermediate nodes.
G.add_node("cc")
G.add_edge("dep", "cc")
G.add_node("conj")
G.add_edge("dep", "conj")
G.add_node("expl")
G.add_edge("dep", "expl")
# Add mod node.
G.add_node("mod")
G.add_edge("dep", "mod")
G.add_node("amod")
G.add_edge("mod", "amod")
G.add_node("appos")
G.add_edge("mod", "appos")
G.add_node("advcl")
G.add_edge("mod", "advcl")
G.add_node("det")
G.add_edge("mod", "det")
G.add_node("acl")
G.add_edge("mod", "acl")
G.add_node("preconj")
G.add_edge("mod", "preconj")
G.add_node("mwe")
G.add_edge("mod", "mwe")
G.add_node("mark")
G.add_edge("mwe", "mark")
G.add_node("advmod")
G.add_edge("mod", "advmod")
G.add_node("neg")
G.add_edge("advmod", "neg")
G.add_node("relcl")
G.add_edge("mod", "relcl")
G.add_node("quantmod")
G.add_edge("mod", "quantmod")
G.add_node("compound")
G.add_edge("mod", "compound")
G.add_node("nn")
G.add_edge("mod", "nn")
G.add_node("nounmod")
G.add_edge("mod", "nounmod")
G.add_node("npmod")
G.add_edge("mod", "npmod")
G.add_node("nummod")
G.add_edge("mod", "nummod")
G.add_node("prep")
G.add_edge("mod", "prep")
G.add_node("poss")
G.add_edge("mod", "poss")
G.add_node("case")
G.add_edge("mod", "case")
# # Add remaining nodes.
G.add_node("parataxis")
G.add_edge("dep", "parataxis")
G.add_node("intj")
G.add_edge("dep", "intj")
G.add_node("prt")
G.add_edge("dep", "prt")
G.add_node("punct")
G.add_edge("dep", "punct")

# Visualize to confirm it is the right configuration.
fig, ax = plt.subplots(figsize=(180, 15))
ax.set_autoscale_on(True)
pos = hierarchy_pos(G,"root")
nx.draw(G, pos=pos, with_labels=True, font_size=6, node_size=300, node_color='w')
# plt.savefig('hierarchy.png')

# For all pairs of dependencies, calculate the path length.
paths = []
for tag1 in tags:
    for tag2 in tags:
        paths.append(len(nx.shortest_path(G, tag1, tag2)))

# Normalize these lengths by the maximum hierarchy depth.
max_depth = max(paths)
similarities = []
for path in paths:
    similarities.append(get_leacock_chodorow_distance(path, max_depth))



# Rescale and convert the LCH score to a distance.
lch_distance = []
mx, mn = max(similarities), min(similarities)
for value in similarities:
    scaled = ((value - mn) / (mx - mn)) * (1 - 0) + 0
    lch_distance.append(1 - scaled)

# Save these distance in a lookup table in an appropriate res directory.
lch_distance_table = {}
for tag1 in tags:
    lch_distance_table[tag1] = {}
    for tag2 in tags:
        lch_distance_table[tag1][tag2] = lch_distance.pop(0)

with open('dependency_distances', "wb") as f:
    dill.dump(lch_distance_table, f)

