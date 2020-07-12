import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.dag import descendants, ancestors
from itertools import combinations


class DirectedAcyclicGraph:
    def __init__(self, exposure, outcome):
        """Constructs a Directed Acyclic Graph (DAG) for determination of adjustment sets

        Parameters
        ----------
        exposure : str
            Exposure of interest in the causal diagram
        outcome : str
            Outcome of interest in the causal diagram

        # TODO add other implementations in the future... have as self.mediator, self.censor, self.missing

        Examples
        --------
        Setting up environment

        >>> from zepid.causal.causalgraph import DirectedAcyclicGraph

        Creating directed acyclic graph

        >>> dag = DirectedAcyclicGraph(exposure="X", outcome="Y")
        >>> dag.add_arrow(source="X", endpoint="Y")
        >>> dag.add_arrow(source="V", endpoint="Y")
        >>> dag.add_arrows(pairs=(("W", "X"), ("W", "Y")))

        Determining adjustment sets

        >>> dag.calculate_adjustment_sets()
        >>> dag.adjustment_sets
        >>> dag.minimal_adjustment_sets

        Plot diagram

        >>> dag.draw_dag()
        >>> plt.show()

        Assess arrow misdirections that result in the chosen adjustment set being invalid

        >>> dag.assess_misdirections(chosen_adjustment_set=set("W"))

        References
        ----------
        Shrier I, & Platt RW. (2008). Reducing bias through directed acyclic graphs.
        BMC medical research methodology, 8(1), 70.
        """
        self.exposure = exposure
        self.outcome = outcome

        dag = nx.DiGraph()
        dag.add_edge(self.exposure, self.outcome)
        self.dag = dag

        self.adjustment_sets = None
        self.minimal_adjustment_sets = None
        self.arrow_misdirections = None

    def add_arrow(self, source, endpoint):
        """Add a single arrow to the current causal DAG

        Parameters
        ----------
        source : str
            Node that arrow originates from
        endpoint : str
            Node that arrow points to
        """
        dag = self.dag.copy()
        dag.add_edge(source, endpoint)
        if not nx.is_directed_acyclic_graph(dag):
            raise DAGError("Cyclic graph detected. Invalid addition for arrow.")

        self.dag = dag

    def add_arrows(self, pairs):
        """Add a set of arrows to the current causal DAG

        Parameters
        ----------
        pairs : list, set, container
            Set of sets of node pairs to add arrows in the DAG, with the first node as the source (node that arrows
             originates from) and the second node is the endpoint (node that arrow points to)
        """
        dag = self.dag.copy()
        dag.add_edges_from(pairs)
        if not nx.is_directed_acyclic_graph(dag):
            raise DAGError("Cyclic graph detected. Invalid addition for arrow(s).")

        self.dag = dag

    def add_from_networkx(self, network):
        # Checking that it is a directed acyclic graph
        if not nx.is_directed_acyclic_graph(network):
            raise DAGError("Cyclic graph detected. Invalid networkx input.")

        # Checking that exposure and outcome are valid nodes
        nodes = list(network.nodes)
        if self.exposure not in nodes:
            raise DAGError(str(self.exposure)+" is not a node in the DAG")
        if self.outcome not in nodes:
            raise DAGError(str(self.outcome)+" is not a node in the DAG")

        self.dag = network.copy()

    def draw_dag(self, positions=None, invert=False, fig_size=(6, 5), node_size=1000):
        """Draws the current input causal DAG

        Parameters
        ----------
        positions :
            Option to provide node locations based on other functionalities
        invert : bool, optional
            To display the (often hidden) assumptions of causal DAG, the invert options displays all arrows assumed
            to not exist. The reversal is an easy way to display lower densities of in the original DAG. No direction
            is provided for these arrows
        fig_size : set
            controls the figure size returned
        node_size : int, float
            Size of nodes in the subsequent plot
        """
        if invert:
            dag = nx.complement(self.dag.to_undirected())
        else:
            dag = self.dag.copy()

        fig = plt.figure(figsize=fig_size)
        ax = plt.subplot(1, 1, 1)

        if positions is None:
            positions = nx.spectral_layout(self.dag)

        nx.draw_networkx(dag, positions, node_color="#d3d3d3", node_size=node_size,
                         edge_color='black', linewidths=1.0, width=1.5,
                         arrowsize=15, ax=ax, font_size=12)
        plt.axis('off')
        return ax

    def calculate_adjustment_sets(self):
        """Determines all sufficient adjustment sets for the causal diagram using the algorithm described in Shrier &
        Platt "Reducing bias through directed acyclic graphs" BMC Medical Research Methodology 2008.

        All possible adjustment sets are enumerated and then assessed. We can briefly consider this as a backtracking
        algorithm where we assess each possible combination that exists within the data

        # TODO in future should allow for adjustment sets to determine causal, censor, missing sets. default to all

        Adjustment sets are added as `DirectedAcyclicGraph.adjustment_sets` and
        `DirectedAcyclicGraph.minimal_adjustment_sets`
        """
        # Extracting list of all sets to check
        sets_to_check = self._define_all_adjustment_sets_(dag=self.dag)

        valid_sets = []
        for adj_set in sets_to_check:
            if self._check_valid_adjustment_set_(graph=self.dag, adjustment_set=adj_set):
                valid_sets.append(adj_set)

        self.adjustment_sets = valid_sets
        self.minimal_adjustment_sets = [x for x in valid_sets if len(x) == len(min(valid_sets, key=len))]

    def _define_all_adjustment_sets_(self, dag):
        """Background function to determine all possible adjustment set combinations to explore. Used to explore every
        possible combinations of adjustment sets to assess whether they are valid for d-separation.
        """
        # List of all nodes valid for adjustment
        all_nodes = list(dag.nodes)
        all_nodes.remove(self.exposure)
        all_nodes.remove(self.outcome)
        list_of_sets = []
        for i in range(0, len(all_nodes) + 1):
            list_of_sets.extend([x for x in combinations(all_nodes, i)])
        return list_of_sets

    def _check_valid_adjustment_set_(self, graph, adjustment_set):
        """Checks the adjustment set as valid using the following 6 steps
        Step 1) check no descendants of X are included in adjustment set
        Step 2) delete variables that meet certain definitions
        Step 3) delete all arrows that originate from exposure
        Step 4) connect all source nodes (to assess for collider stratification)
        Step 5) convert to undirected graph
        Step 6) check whether a path exists between exposure & outcome
        """
        dag = graph.copy()

        # List of all nodes valid for adjustment
        all_nodes = list(dag.nodes())
        all_nodes.remove(self.exposure)
        all_nodes.remove(self.outcome)

        # Step 1) Check no descendants of X
        desc_x = descendants(dag, self.exposure)
        if desc_x & set(adjustment_set):
            return False

        # Step 2) Delete all variables that: (a) non-ancestors of X, (b) non-ancestors of Y, (c) non-ancestors
        #         of adjustment set
        set_check = set(adjustment_set).union([self.exposure, self.outcome])
        set_remove = set(dag.nodes)
        for n in set_check:
            set_remove = set_remove & (dag.nodes - ancestors(dag, n))
        set_remove = set_remove - set([self.exposure, self.outcome]) - set(adjustment_set)
        dag.remove_nodes_from(set_remove)

        # Step 3) Delete all arrows with X as the source
        for endpoint in list(dag.successors(self.exposure)):
            dag.remove_edge(self.exposure, endpoint)

        # Step 4) Directly connect all source nodes pointing to same endpoint (for collider assessment)
        for n in dag:
            sources = list(dag.predecessors(n))
            if len(sources) > 1:
                for s1, s2 in combinations(sources, 2):
                    if not (dag.has_edge(s2, s1) or dag.has_edge(s1, s2)):
                        dag.add_edge(s1, s2)

        # Step 5) Remove arrow directionality
        uag = dag.to_undirected()

        # Step 6) Remove nodes from the adjustment set
        uag.remove_nodes_from(adjustment_set)

        # Checking whether a a path between X and Y exists now
        if nx.has_path(uag, self.exposure, self.outcome):
            return False
        else:
            return True

    def assess_misdirections(self, chosen_adjustment_set):
        """Arrow direction can potentially be misspecified. This function checks every possible arrow reversal and
        determines the relevant sufficient adjustment sets. Those new adjustment sets are compared to the chosen
        adjustment set for differences.

        Parameters
        ----------
        chosen_adjustment_set : set, list, container
            The sufficient adjustment set selected for the data analysis
        """
        all_edges = list(self.dag.edges())
        l = []
        for i in range(0, len(all_edges)+1):
            l.append([x for x in combinations(all_edges, i)])

        valid_switches = []
        valid_graphs = []
        for c in range(1, len(l)):
            for s in l[c]:
                # Copy graph
                g = self.dag.copy()
                g.remove_edges_from(s)  # Remove edge
                # Reversing all edges in that set
                for pair in s:
                    g.add_edge(pair[1], pair[0])  # Add reversed edge
                # Check if DAG
                if nx.is_directed_acyclic_graph(g):
                    valid_graphs.append(g)
                    valid_switches.append(s)

        alternative_adjustment_sets = {}
        for v, g in zip(valid_switches, valid_graphs):
            sets_to_check = self._define_all_adjustment_sets_(dag=g)
            valid_sets = []
            for adj_set in sets_to_check:
                if self._check_valid_adjustment_set_(graph=g, adjustment_set=adj_set):
                    valid_sets.append(adj_set)
            if chosen_adjustment_set not in set(valid_sets):
                alternative_adjustment_sets[v] = valid_sets

        # print(alternative_adjustment_sets)
        self.arrow_misdirections = alternative_adjustment_sets


class DAGError(Exception):
    """Exception raised for errors in Directed Acyclic Graphs not being directed or acyclic
    """

    def __init__(self, message):
        super().__init__(message)
