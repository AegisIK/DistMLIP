from matgl.layers import CHGNetBondGraphBlock, CHGNetLineGraphConv
from torch import Tensor
import dgl


class CHGNetBondGraphBlock_Dist(CHGNetBondGraphBlock):
    @classmethod
    def from_existing(cls, existing):
        assert isinstance(existing, CHGNetBondGraphBlock), f"from_existing only converts CHGNetBondGraphBlock, not {type(existing)}"
        dist_block = cls.__new__(cls)
        dist_block.__dict__ = existing.__dict__.copy()
        dist_block.conv_layer = CHGNetLineGraphConv_Dist.from_existing(dist_block.conv_layer)

        return dist_block

    def forward(
        self,
        graph: dgl.DGLGraph,
        atom_features: Tensor,
        bond_features: Tensor,
        angle_features: Tensor,
        shared_node_weights: Tensor | None,
        compute_dist: bool = False,
        convolution_type: str = "both"
    ) -> tuple[Tensor, Tensor]:
        """Perform convolution in BondGraph to update bond and angle features.

        Args:
            graph: bond graph (line graph of atom graph)
            atom_features: atom features
            bond_features: bond features
            angle_features: concatenated center atom and angle features
            shared_node_weights: shared node message weights
            compute_dist: whether or not this convolution is occurring in distributed mode
            convolution_type: type of convolution ("node" vs "edge" vs "both")

        Returns:
            tuple: update bond features, update angle features
        """
        if compute_dist:
            node_features = bond_features
        else:
            node_features = bond_features[graph.ndata["bond_index"]]

        edge_features = angle_features
        aux_edge_features = atom_features[graph.edata["center_atom_index"]]

        bond_features_, angle_features = self.conv_layer(
            graph, node_features, edge_features, aux_edge_features, shared_node_weights, convolution_type=convolution_type
        )

        bond_features_ = self.bond_dropout(bond_features_)
        angle_features = self.angle_dropout(angle_features)

        if not compute_dist:
            bond_features[graph.ndata["bond_index"]] = bond_features_
        else:
            bond_features = bond_features_

        return bond_features, angle_features


class CHGNetLineGraphConv_Dist(CHGNetLineGraphConv):
    
    @classmethod
    def from_existing(cls, existing):
        assert isinstance(existing, CHGNetLineGraphConv), f"from_existing only converts CHGNetLineGraphConv, not {type(existing)}"
        dist_conv = cls.__new__(cls)
        dist_conv.__dict__ = existing.__dict__.copy()

        return dist_conv

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_features: Tensor,
        edge_features: Tensor,
        aux_edge_features: Tensor,
        shared_node_weights: Tensor | None,
        convolution_type: str = "both"
    ) -> tuple[Tensor, Tensor]:
        """Perform sequence of edge->node->states updates.

        Args:
            graph: bond graph (line graph of atom graph)
            node_features: bond features (edge features (for bonds within three body cutoff in atom graph)
            edge_features: angle features (edge features to be updated)
            aux_edge_features: center atom features (edge features that are not updated)

            shared_node_weights: shared node message weights

        Returns:
            tuple: update edge features, update node features
            note that the node features are the bond features included in the line graph only.
        """
        with graph.local_scope():
            graph.ndata["features"] = node_features
            graph.edata["features"] = edge_features
            graph.edata["aux_features"] = aux_edge_features

            if convolution_type == "both" or convolution_type == "node":
                # node (bond) update
                node_update = self.node_update_(graph, shared_node_weights)
                new_node_features = node_features + node_update
                graph.ndata["features"] = new_node_features
            else:
                new_node_features = node_features

            if convolution_type == "both" or convolution_type == "edge":
                # edge (angle) update (should angle update be done before node update?)
                if self.edge_update_func is not None:
                    edge_update = self.edge_update_(graph)
                    new_edge_features = edge_features + edge_update
                    graph.edata["features"] = new_edge_features
                else:
                    new_edge_features = edge_features
            else:
                new_edge_features = edge_features

        return new_node_features, new_edge_features