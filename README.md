# Cactus Protocol

Cactus is a decentralized AI inference protocol that enables efficient and secure execution of AI tasks through a network of specialized nodes (orchestrators) and routing servers. The protocol ensures fair compensation, quality verification, and reliable task execution.


## 🚀 Before Running the Client

Make sure the following dependencies are installed:

---

### 🐍 Python 3.13.2  
🔗 [Download Python 3.13.2](https://www.python.org/downloads/release/python-3132/)

---

### 🦀 Rust  
🔗 [Install Rust](https://www.rust-lang.org/tools/install)

---

### ⚙️ Install CUDA-enabled PyTorch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

### 📦 Install Python Requirements

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Client

```bash
cactus.exe client --server-url SERVER_URL --public-ip NODE_IP --port NODE_PORT
```

### 🧠 Parameters:

- `SERVER_URL` – The Gateway URL.  
  **Default:** `http://3.110.255.211:8001` (if not provided)
- `NODE_IP` – The IP address of your client machine.  
  This should be **static** so jobs can reach you.
- `NODE_PORT` – The port on which the client will run on your node.

---

## 🌐 Running on Local Machine with EC2 Tunneling

If you’re running the client on a **local PC** and want to expose your `NODE_PORT` to an EC2 instance, use SSH port forwarding:

```bash
ssh -i <your_key.pem> -4 -R 0.0.0.0:<EC2_PORT>:127.0.0.1:<NODE_PORT> ec2-user@<EC2_PUBLIC_IP>
```

### ✅ Example:

```bash
ssh -v -i cactus.pem -4 -R 0.0.0.0:8002:127.0.0.1:8002 ec2-user@3.110.255.211
```


## Architecture Overview

### Core Components

1. **Routing Servers**
   - Act as the entry point for client requests
   - Analyze available orchestrators and match tasks to appropriate nodes
   - Verify task completion and manage payment distribution
   - Maintain orchestrator reputation and performance metrics

2. **Orchestrators**
   - Specialized nodes that execute AI inference tasks
   - Must stake tokens to participate in the network
   - Can delegate tasks to other orchestrators
   - Track performance metrics (successful tasks, compute time)

3. **Clients**
   - End users requesting AI inference services
   - Submit tasks through routing servers
   - Receive results and pay for completed tasks

### Directory Structure

```
protocol/
├── rust/
│   └── src/
│       └── protocol/
│           ├── mod.rs      # Protocol module entry point
│           └── types.rs    # Core data structures and types
└── contracts/
    ├── sources/
    │   ├── cactus_payment.move    # Smart contract for payment and task management
    │   └── cactus_payment_tests.move  # Contract tests
    └── Move.toml           # Move package configuration
```

## Data Flow

1. **Task Creation**
   ```
   Client -> Routing Server -> Smart Contract
   ```
   - Client submits task request to routing server
   - Server analyzes orchestrator availability and capabilities
   - Server creates task on smart contract with assigned orchestrator

2. **Task Execution**
   ```
   Routing Server -> Orchestrator -> Smart Contract
   ```
   - Server forwards task to selected orchestrator
   - Orchestrator executes AI inference
   - Results are verified by the server

3. **Task Completion**
   ```
   Orchestrator -> Routing Server -> Smart Contract -> Client
   ```
   - Orchestrator submits results to server
   - Server verifies quality and marks task complete
   - Payment is released to orchestrator
   - Results are sent to client

## Logic Flow

### Task Lifecycle

1. **Initialization**
   - Orchestrators register with stake
   - Servers register and get authorized
   - Initial reputation metrics are established

2. **Task Assignment**
   - Server analyzes orchestrator capabilities
   - Considers factors like:
     - Current stake amount
     - Success rate
     - Compute time history
     - Network latency

3. **Execution**
   - Orchestrator receives task details
   - Executes AI inference
   - Can delegate to other orchestrators if needed
   - Tracks compute time and resources

4. **Verification**
   - Server verifies result quality
   - Checks against task requirements
   - Updates orchestrator metrics

5. **Payment**
   - Smart contract handles payment distribution
   - Calculates rewards based on:
     - Task complexity
     - Compute time
     - Success rate
   - Updates orchestrator stake

### Security Model

1. **Stake-based Security**
   - Orchestrators must stake tokens
   - Higher stake = higher trust and priority
   - Malicious behavior can result in stake slashing

2. **Server Authorization**
   - Only authorized servers can create tasks
   - Servers must verify task completion
   - Prevents unauthorized task creation

3. **Delegation Security**
   - Orchestrators can delegate tasks
   - Delegation relationships are tracked
   - Original orchestrator remains responsible

## Smart Contract Features

The `cactus_payment.move` contract implements:

1. **Task Management**
   - Task creation and assignment
   - Status tracking
   - Result verification

2. **Payment System**
   - Stake management
   - Payment distribution
   - Reward calculation

3. **Orchestrator Management**
   - Registration and stake
   - Performance tracking
   - Delegation handling

4. **Server Management**
   - Authorization
   - Task verification
   - Quality control

## Getting Started

1. **Setup Development Environment**
   ```bash
   # Install Aptos CLI
   curl -fsSL https://raw.githubusercontent.com/aptos-labs/aptos-core/main/scripts/dev_setup.sh | bash

   # Install Rust
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Build and Test**
   ```bash
   # Compile Move contracts
   cd contracts
   aptos move compile

   # Run tests
   aptos move test

   # Build Rust protocol
   cd ../protocol/rust
   cargo build
   ```

3. **Deploy**
   ```bash
   # Deploy to devnet
   cd ../contracts
   aptos move publish --network devnet
   ```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 

