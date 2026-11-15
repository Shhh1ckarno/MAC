from .auction import AuctionCoordinator
from .base import Coordinator
from .central import CentralCoordinator
from .cnp import CNPCoordinator
from .drone_cnp import PassiveCoordinator
from .hungarian import hungarian


def make_coordinator(mechanism: str) -> Coordinator:
    if mechanism == "central_greedy":
        return CentralCoordinator("greedy")
    if mechanism == "central_hungarian":
        return CentralCoordinator("hungarian")
    if mechanism == "cnp":
        return CNPCoordinator()
    if mechanism == "auction_first":
        return AuctionCoordinator("first")
    if mechanism == "auction_vickrey":
        return AuctionCoordinator("vickrey")
    if mechanism == "cnp_drone":
        return PassiveCoordinator()
    raise ValueError(mechanism)


__all__ = ["AuctionCoordinator", "CNPCoordinator", "CentralCoordinator", "Coordinator", "hungarian", "make_coordinator"]
