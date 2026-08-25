# Review summary

This branch adds no approximation to the production solver. It adds an external, fail-closed campaign runner around the existing screened and exact execution modes.

Review focus:

- screen and exact environments are disjoint;
- exact outputs never reuse screen files;
- duplicate input paths and batch width >64 fail closed;
- screen receipts must declare approximation and exact-rerun obligation;
- all survivor exact outputs and validators must pass before `valid=true`;
- the MASTER W16 speedup includes screen and exact wall;
- exact throughput and effective candidate throughput are not conflated.
