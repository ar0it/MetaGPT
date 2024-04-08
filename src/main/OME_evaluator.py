import xml.etree.ElementTree as ET
import javabridge
import bioformats
import numpy as np
from zss import simple_distance, Node
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# javabridge.start_vm(class_path=bioformats.JARS)


class OMEEvaluator:
    """
    This class evaluates the performance of a OME XML generation model by calculating the edit distance between the
    ground truth and the prediction. https://github.com/timtadh/zhang-shasha
    """

    def __init__(self, path_to_raw_metadata=None,
                 gt_path="/home/aaron/PycharmProjects/MetaGPT/raw_data/testetst_Image8_edited_.ome.xml",
                 pred_path="/home/aaron/PycharmProjects/MetaGPT/raw_data/image8_start_point.ome.xml"):
        """
        :param path_to_raw_metadata: path to the raw metadata file
        """
        self.gt_path = gt_path
        self.pred_path = pred_path

        self.prediction = self.read_ome_xml(pred_path)
        self.ground_truth = self.read_ome_xml(gt_path)
        self.edit_score = self.edit_distance(self.prediction, self.ground_truth)
        self.report()

    def edit_distance(self, xml_a, xml_b):
        """
        Calculate the edit distance between two xml trees on word level.
        Here an outline of the algorithm:
        1. Get the paths of the xml trees.
        2. Align the paths such that the distance between the paths is minimized.
        3. Calculate the word level edit distance between the paths.
        """
        # get the paths of the xml trees

        self.pred_graph = self.get_graph(self.prediction)
        self.gt_graph = self.get_graph(self.ground_truth)
        return simple_distance(self.pred_graph, self.gt_graph)

    def align_paths(self, paths_a, paths_b):
        """
        Align the paths such that the sum of distances between the paths is minimized.
        paths_a: set of paths
        paths_b: set of paths
        :return: list of tuples of aligned paths
        """
        print("- - - Aligning paths - - -")
        # Assuming A and B are your lists and f is your function

        # Create the matrix
        matrix = [[self.align_sequences_score(a, b) for b in paths_b] for a in paths_a]
        print("matrix", [print(x, "\n") for x in matrix])

        # score, alignment_a, alignment_b = self.align_sequences(paths_a, paths_b, cost=self.align_sequences_score)
        return 1, 1, 1

    def align_sequences_score(self, s1, s2, cost=lambda a, b: a != b):
        """
        returns only the score for the alignment
        :param s1:
        :param s2:
        :param cost:
        :return:
        """
        score, alignment_a, alignment_b = self.align_sequences(s1, s2, cost=lambda a, b: a != b)
        return score

    def align_sequences(self, s1, s2, cost=lambda a, b: a != b):
        print("- - - Aligning sequences - - -")
        m, n = len(s1), len(s2)
        # Create a matrix to store the cost of edits
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        dp = np.zeros((m + 1, n + 1))

        # Initialize the matrix with the cost of deletions and insertions
        for i in range(m + 1):
            dp[i][0] = i * +1
        for j in range(n + 1):
            dp[0][j] = j * +1

        # Fill the matrix based on the cost of substitution, insertion, and deletion
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]  # No cost if characters are the same
                else:
                    dp[i][j] = min(dp[i - 1][j - 1] + cost(s1[i - 1], s2[j - 1]),  # Substitution
                                   dp[i - 1][j] + cost(s1[i - 1], s2[j - 1]),  # Deletion
                                   dp[i][j - 1] + cost(s1[i - 1], s2[j - 1]),  # Insertion
                                   )
        # Reconstruct the alignment
        alignment_a, alignment_b = [], []
        i, j = m, n
        while i > 0 and j > 0:
            if s1[i - 1] == s2[j - 1]:
                alignment_a = [s1[i - 1]] + alignment_a
                alignment_b = [s2[j - 1]] + alignment_b
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j - 1] + cost(s1[i - 1], s2[j - 1]):  # substitution
                alignment_a = [s1[i - 1]] + alignment_a
                alignment_b = [s2[j - 1]] + alignment_b
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j] + cost(s1[i - 1], s2[j - 1]):  # deletion
                alignment_a = [s1[i - 1]] + alignment_a
                alignment_b = [["-"]] + alignment_b
                i -= 1
            else:  # dp[i][j] == dp[i][j - 1] + cost_insertion  # insertion
                alignment_a = [["-"]] + alignment_a
                alignment_b = [s2[j - 1]] + alignment_b
                j -= 1

        # Handle the remaining characters in s1 or s2
        while i > 0:
            alignment_a = [s1[i - 1]] + alignment_a
            alignment_b = [[["-"] * len(s1[j - 1])]] + alignment_b
            i -= 1
        while j > 0:
            alignment_a = [[["-"] * len(s2[j - 1])]] + alignment_a
            alignment_b = [s2[j - 1]] + alignment_b
            j -= 1
        print(dp)
        return dp[-1][-1], alignment_a, alignment_b

    def word_edit_distance(self, aligned_paths) -> int:
        """
        Calculate the word level edit distance between two sets of paths.
        aligned_paths: list of tuples of aligned paths
        """
        edit_distance = 0
        for path_a, path_b in aligned_paths:
            # set the number of iterations to the length of the longest path
            iterations = len(path_a) if len(path_a) > len(path_b) else len(path_b)
            distance = 0
            for i in range(iterations):
                node_b = path_b[i] if len(path_b) > i else None
                node_a = path_a[i] if len(path_a) > i else None
                if node_a != node_b:
                    distance += 1
            edit_distance += distance

        return edit_distance

    def get_paths(self, xml_root, path: str = '') -> set:
        """
        Helper function to get all paths in an XML tree.
        :return: set of paths
        """
        paths = set()
        for child in xml_root:
            new_path = path + '/' + child.tag.split('}')[1]
            if child.attrib:
                for key in child.attrib.keys():
                    paths.add(new_path + '/' + key + '=' + child.attrib[key])
                    paths.update(self.get_paths(child, new_path))
            else:
                paths.add(new_path)
                paths.update(self.get_paths(child, new_path))
        return paths

    def get_graph(self, xml_root, root=None):
        """
        Helper function to get the graph representation of an XML tree.
        """
        if root is None:
            root = Node("OME")
            if xml_root.attrib:
                for key in xml_root.attrib.keys():
                    root.addkid(Node(key + '=' + xml_root.attrib[key]))

        for child in xml_root:
            new_node = Node(child.tag.split('}')[1])
            root.addkid(new_node)
            if child.attrib:
                for key in child.attrib.keys():
                    new_node.addkid(Node(key + '=' + child.attrib[key]))
            self.get_graph(child, new_node)
        return root

    def path_difference(self, xml_a, xml_b):
        """
        Calculates the length of the difference between the path sets in two xml trees.
        """
        paths_a = self.get_paths(xml_a)
        paths_b = self.get_paths(xml_b)
        print("path difference: ", paths_a)
        return len(paths_a.symmetric_difference(paths_b))

    def read_ome_xml(self, path):
        """
        This method reads the ome xml file and returns the root element.
        """
        tree = ET.parse(path)
        root = tree.getroot()
        print("root", tree)
        return root

    def read_ome_tiff(self, path):
        """
        This method reads the ome tiff file.
        """
        ome = bioformats.OMEXML()
        print(ome)
        return ome

    def evaluate(self):
        """
        compare the two ome xml trees and return the similarity score by calculating the edit distance between the two xml.
        """
        print("Evaluation in progress...")
        self.score = self.path_difference(self.prediction, self.ground_truth)

    def report(self):
        """
        Write evaluation report to file.
        """
        with open(f"../../out/report_{self.gt_path.split('/')[-1]}.md", "w") as f:
            f.write("# Evaluation Report\n")
            f.write("## File content\n")
            f.write(f"### File 1: {self.gt_path.split('/')[-1]}\n")
            paths_gt = self.get_paths(self.ground_truth)
            f.write(f"Number of paths: {len(paths_gt)}\n")  # TODO: Somehow incorporate the ID
            f.write(f"Number of paths in structured annotations:"
                    f"{len([x for x in paths_gt if x.__contains__('StructuredAnnotations')])}\n")
            f.write(f"### File 2: {self.pred_path.split('/')[-1]}\n")
            paths_pred = self.get_paths(self.prediction)
            f.write(f"Number of paths: {len(paths_pred)}\n")
            f.write(f"Number of paths in structured annotations: "
                    f"{len([x for x in paths_pred if x.__contains__('StructuredAnnotations')])}\n")
            f.write("## Edit Distance\n")
            f.write(f"Edit distance between Files: {self.edit_score}\n")
            # create a pandas dataframe with all the paths
            df_paths = pd.DataFrame(set(paths_gt.union(paths_pred)))
            df_paths["GT"] = [1 if x in paths_gt else 0 for x in df_paths[0]]
            df_paths["Pred"] = [1 if x in paths_pred else 0 for x in df_paths[0]]
            df_paths["StructuredAnnotations"] = [1 if x.__contains__("StructuredAnnotations") else 0 for x in df_paths[0]]

            plt.bar([1, 2], [df_paths["GT"].sum(), df_paths["Pred"].sum()])
            plt.bar([1, 2], [df_paths["GT"]
                             [df_paths["StructuredAnnotations"] == 1].sum(), df_paths["Pred"]
            [df_paths["StructuredAnnotations"] == 1].sum()])

            plt.title("Path Comparison")
            plt.xticks([1, 2], ["GT", "Pred"])
            plt.legend(title="Annotation")

            print(df_paths)
            plt.savefig(f"../../out/paths_{self.gt_path.split('/')[-1]}.png")
            # add the plot to the report
            f.write("## Path Comparison\n")
            f.write(f"![barplot comparing the two xml files](paths_{self.gt_path.split('/')[-1]}.png)\n")






# javabridge.kill_vm()

if __name__ == "__main__":
    evaluator = OMEEvaluator()
