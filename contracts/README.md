# Cactus Payment Smart Contract

This directory contains the Move smart contract for the Cactus payment system on Aptos blockchain.

## Overview

The Cactus payment contract manages orchestrator registration, task creation, completion, and delegation. It includes a secure delegation pattern for orchestrator operations.

## Project Structure

```
contracts/
├── build/                # Compiled bytecode and build artifacts
├── sources/              # Smart contract source files
│   ├── cactus_payment.move       # Main contract implementation
│   └── cactus_payment_tests.move # Unit tests for the contract
└── Move.toml             # Package definition and dependencies
```

## Architecture

The system follows a server-client architecture where:

1. **Routing Server**: A central server that manages client requests and orchestrator selection
   - Maintains a list of available orchestrators
   - Routes client requests to appropriate orchestrators
   - Handles blockchain interactions on behalf of clients
   - Is solely responsible for verifying and completing tasks
   
2. **Orchestrators**: Compute providers that execute tasks
   - Register with stake to guarantee service
   - Can delegate operational tasks to other addresses (for operations other than task completion)
   - Execute assigned tasks and return results to the server
   - Receive payment when the server validates and completes their tasks

3. **Clients**: End users requesting compute services
   - Don't need blockchain accounts
   - Submit requests through the routing server
   - Are identified by unique identifiers provided by the server

## Task Lifecycle

1. **Task Creation**: The server selects an appropriate orchestrator for a client's request and creates a task on-chain.
2. **Task Execution**: The selected orchestrator processes the task off-chain and returns the result to the server.
3. **Task Completion**: The server (and only the server that created the task) verifies the result and completes the task on-chain, which releases payment to the orchestrator.

## Key Features

- **Server-Based Task Management:** Servers create and complete tasks on behalf of clients, maintaining full control over the task lifecycle
- **Orchestrator Registration:** Register as an orchestrator with a stake amount
- **Task Management:** Create and complete computational tasks
- **Secure Delegation:** Allow other addresses to act on behalf of an orchestrator for specific operations
- **Statistics Tracking:** Track orchestrator performance metrics

## Implementation Details

### Data Structures

The contract uses several key data structures:

1. **Task**: Stores information about a computational task
   - Client address (server that created it)
   - Client identifier (for the actual end-user)
   - Orchestrator address
   - Compute time and GPU resources used
   - Model type
   - Payment amount
   - Result hash
   - Status (pending, completed, failed, disputed)

2. **OrchestratorStats**: Tracks performance metrics for orchestrators
   - Stake amount
   - Successful tasks completed
   - Total compute time used

3. **DelegatedOrchestrator**: Manages delegation relationships
   - Owner address
   - List of delegated addresses

### Key Operations

1. **Server Authorization**: Only the admin account can register authorized servers
2. **Task Creation**: Authorized servers can create tasks on behalf of clients
3. **Task Completion**: Only the server that created a task can mark it as complete
4. **Delegation**: Orchestrators can delegate operations to other addresses
5. **Statistics Tracking**: Orchestrator performance metrics are updated automatically

## Security Model

1. **Server Authorization:**
   - Only authorized servers can create tasks
   - Only the specific server that created a task can complete it
   - Server registration is controlled by the contract administrator
   
2. **Orchestrator Validation:**
   - Orchestrators must stake funds to participate
   - The server ensures that only the correct orchestrator receives payment
   - Payment is only released after successful task completion and verification

3. **Delegation Security:**
   - Orchestrators can securely delegate operations to other addresses (except task completion)
   - Delegation is verified on-chain for all operations

4. **End-to-End Verification:**
   - The server creates tasks with specific orchestrators
   - The server verifies results before marking tasks complete
   - The server ensures payment goes to the correct orchestrator
   - Task result hashes are stored on-chain for verification

## Testing

The contract has extensive test coverage (95.85%), covering all functional aspects of the code.

### Running Tests

To run the tests:

```bash
cd /path/to/contracts
aptos move test
```

### Test Coverage

To see test coverage:

```bash
aptos move test --coverage
```

For detailed line-by-line coverage:

```bash
aptos move coverage source --module payment
```

## Current Test Coverage: 95.85%

The only uncovered code is initialization functions that are only called during module publication to the blockchain and cannot be directly tested in unit tests.

## Test Cases

The test suite includes:

1. **Server Management**
   - Register authorized servers
   - Prevent unauthorized server registration
   - Prevent duplicate server registration

2. **Orchestrator Registration**
   - Register an orchestrator with stake
   - Attempt to register already registered orchestrator

3. **Task Management**
   - Create a task through an authorized server
   - Complete a task with the correct server
   - Try to complete a task with the wrong server (should fail)
   - Try to complete a task with an unauthorized server (should fail)
   - Try to create a task with an unauthorized server (should fail)
   - Try to complete a task for the wrong orchestrator (should fail)
   - Try to complete a task that doesn't exist (should fail)
   - Try to complete a task that's already completed (should fail)

4. **Delegation**
   - Add and verify delegated orchestrators
   - Test delegation with multiple orchestrators

5. **Helper Functions**
   - Get and verify orchestrator stats
   - Verify task result hash retrieval
   - Test failure cases for missing orchestrators 