#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2020 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
# Mean Approaches
from ensmic.ensemble.mean_unweighted import ELM_MeanUnweighted
from ensmic.ensemble.mean_weighted import ELM_MeanWeighted
# Majority Voting Approaches
from ensmic.ensemble.majorityvote_hard import ELM_MajorityVote_Hard
from ensmic.ensemble.majorityvote_soft import ELM_MajorityVote_Soft
# Machine Learning Approaches
from ensmic.ensemble.decision_tree import ELM_DecisionTree
fro