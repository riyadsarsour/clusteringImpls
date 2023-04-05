# Submit this file to Gradescope
from typing import Dict, List, Tuple
from math import comb, log, log2
import math
# you may use other Python standard libraries, but not data
# science libraries, such as numpy, scikit-learn, etc.


class Solution:

    def dict_to_array(self, confusion_cluster):
        max_row = -1
        max_col = -1
        for i in confusion_cluster.keys():
            if i[0] > max_row:
                max_row = i[0]
            if i[1] > max_col:
                max_col = i[1]

        array = [[0] * (max_col + 1) for _ in range(max_row + 1)]
        for key, value in confusion_cluster.items():
            row, col = key
            try:
                array[row][col] = value
            except:
                continue
        return array

    def true_postive_calc(self, confusion_cluster: Dict[Tuple[int, int], int]) -> int:
        ans = 0
        for i in confusion_cluster.keys():
            ans += comb(confusion_cluster[i], 2)
        return ans

    def false_positive_calc(self, confusion_cluster_matrix: List[List[int]], true_positive: int) -> int:
        tot = 0
        for row in confusion_cluster_matrix:
            row_sum = sum(row)
            tot += comb(row_sum, 2)
        return tot - true_positive

    def true_negative_calc(self, n: int, true_positive: int, false_postive: int, false_negative: int) -> int:
        possible = comb(n, 2)
        return possible - true_positive - false_postive - false_negative

    def false_negative_calc(self, confusion_cluster_matrix: List[List[int]], true_positive: int) -> int:
        sum = 0
        for j in range(len(confusion_cluster_matrix[0])):
            col_sum = 0
            for i in range(len(confusion_cluster_matrix)):
                col_sum += confusion_cluster_matrix[i][j]
            sum += comb(col_sum, 2)

        return sum - true_positive

    def confusion_matrix(self, true_labels: List[int], pred_labels: List[int]) -> Dict[Tuple[int, int], int]:
        """Calculate the confusion matrix and return it as a sparse matrix in dictionary form.
        Args:
          true_labels: list of true labels
          pred_labels: list of predicted labels
        Returns:
          A dictionary of (true_label, pred_label): count
        """
        ans = dict()
        for i, j in zip(true_labels, pred_labels):
            key = (i, j)
            ans[key] = ans.get(key, 0) + 1

        return ans

    def jaccard(self, true_labels: List[int], pred_labels: List[int]) -> float:
        """Calculate the Jaccard index.
        Args:
          true_labels: list of true cluster labels
          pred_labels: list of predicted cluster labels
        Returns:
          The Jaccard index. Do NOT round this value.
        """
        cf = self.confusion_matrix(true_labels, pred_labels)

        cf_array = self.dict_to_array(cf)
        tp = self.true_postive_calc(cf)
        fp = self.false_positive_calc(cf_array, tp)
        fn = self.false_negative_calc(cf_array, tp)
        tn = self.true_negative_calc(len(pred_labels), tp, fp, fn)

        return float(tp/(tp+fn+fp))

    def c_entropy(self, n, cf_array):
        tot = 0

        for i in range(len(cf_array)-1):
            c = cf_array[i][len(cf_array[0])-1]
            if c == 0:
                continue
            c = float(c/n)
            cl = math.log2(c)
            tot -= c*cl

        return float(-1.0*tot)

    def g_entropy(self, n, cf_array):
        tot = 0

        for j in range(len(cf_array[0])-1):
            c = cf_array[len(cf_array)-1][j]
            if c == 0:
                continue
            c = float(c/n)
            tot -= float(c * math.log2(c))
        return float(-1.0*tot)

    def nmi(self, true_labels: List[int], pred_labels: List[int]) -> float:
        """Calculate the normalized mutual information.
        Args:
          true_labels: list of true cluster labels
          pred_labels: list of predicted cluster labels
        Returns:
          The normalized mutual information. Do NOT round this value.
        """
        N = len(pred_labels)

        cf = self.confusion_matrix(true_labels, pred_labels)
        cf_array = self.dict_to_array(cf)
        # print(cf_array)
        for row in cf_array:
            row.append(sum(row))
        j_sum = []
        for j in range(len(cf_array[0])-1):
            col_sum = 0
            for i in range(len(cf_array)):
                col_sum += cf_array[i][j]
            j_sum.append(col_sum)

        # now matrix has C sum and G sum
        cf_array.append(j_sum)
        ...  # implement this function

        entropy_c = self.c_entropy(N, cf_array)
        entropy_g = self.g_entropy(N, cf_array)
        nmi_num = 0
        # print(entropy_c)
        # print(entropy_g)
        for row in range(len(cf_array)-1):
            for col in range(len(cf_array[0])-1):
                pij = float(cf_array[row][col]/N)
                if pij == 0:
                    continue
                pc = float(cf_array[row][len(cf_array[0])-1] / N)
                pg = float(cf_array[len(cf_array)-1][col]/N)
                denom = float(pij/(pc*pg))
                nmi_num += pij * log2(denom)

        sq_rt = float(math.sqrt(entropy_c*entropy_g))

        return float(nmi_num/sq_rt)


# print(Solution.nmi(Solution(), [0, 0, 0, 1, 1, 0, 1, 0, 1, 0], [
#       1, 1, 1, 1, 0, 0, 1, 0, 0, 1]))
