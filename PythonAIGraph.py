from graphviz import Digraph

dot = Digraph()

# Postavljanje opcija za čvorove
dot.attr('node', shape='box', style='filled, rounded', color='black', fontname='helvetica')
# Postavljanje opcija za bridove
dot.attr('edge', fontname='helvetica')

# Dodavanje čvorova
dot.node('0', label='age <= 30.5\ngini = 0.778\nsamples = 18\nvalue = [3, 6, 3, 3, 3]\nclass = Classical', fillcolor='#e5fad7')
dot.node('1', label='gender <= 0.5\ngini = 0.75\nsamples = 12\nvalue = [3, 0, 3, 3, 3]\nclass = Acoustic', fillcolor='#ffffff')
dot.node('2', label='age <= 25.5\ngini = 0.5\nsamples = 6\nvalue = [3, 0, 3, 0, 0]\nclass = Acoustic', fillcolor='#ffffff')
dot.node('3', label='gini = 0.0\nsamples = 3\nvalue = [0, 0, 3, 0, 0]\nclass = Dance', fillcolor='#39e5c5')
dot.node('4', label='gini = 0.0\nsamples = 3\nvalue = [3, 0, 0, 0, 0]\nclass = Acoustic', fillcolor='#e58139')
dot.node('5', label='age <= 25.5\ngini = 0.5\nsamples = 6\nvalue = [0, 0, 0, 3, 3]\nclass = HipHop', fillcolor='#ffffff')
dot.node('6', label='gini = 0.0\nsamples = 3\nvalue = [0, 0, 0, 3, 0]\nclass = HipHop', fillcolor='#3c39e5')
dot.node('7', label='gini = 0.0\nsamples = 3\nvalue = [0, 0, 0, 0, 3]\nclass = Jazz', fillcolor='#e539c0')
dot.node('8', label='gini = 0.0\nsamples = 6\nvalue = [0, 6, 0, 0, 0]\nclass = Classical', fillcolor='#7be539')

# Dodavanje bridova
dot.edge('0', '1', label='True', labeldistance='2.5', labelangle='45', headlabel='True')
dot.edge('1', '2')
dot.edge('2', '3')
dot.edge('2', '4')
dot.edge('1', '5')
dot.edge('5', '6')
dot.edge('5', '7')
dot.edge('0', '8', label='False', labeldistance='2.5', labelangle='-45', headlabel='False')

# Prikaži graf
dot.render('tree_graph', format='png', cleanup=True)
