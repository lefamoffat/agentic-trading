# Agentic Trading Documentation

Welcome to the comprehensive documentation for the Agentic Trading platform. This guide will help you navigate all available documentation based on your needs.

## 🚀 Quick Start

**New to the platform?** Start here:

1. **[Getting Started](getting_started.md)** - Installation, setup, and your first trading experiment
2. **[Architecture Overview](architecture.md)** - Core principles and project philosophy
3. **[Technical Overview](technical_overview.md)** - Detailed system architecture and components

## 📚 Complete Documentation Guide

### Core Platform Documentation

| Document                                        | Purpose                                      | Audience                     |
| ----------------------------------------------- | -------------------------------------------- | ---------------------------- |
| **[Getting Started](getting_started.md)**       | Installation, setup, first experiment        | New users, Quick setup       |
| **[Architecture](architecture.md)**             | Core design principles, library roles        | Developers, System designers |
| **[Technical Overview](technical_overview.md)** | Detailed system architecture                 | Developers, Advanced users   |
| **[Development Guide](development.md)**         | Development workflows, testing, contributing | Contributors, Maintainers    |

### System-Specific Guides

| System              | Documentation                                                   | Purpose                                  |
| ------------------- | --------------------------------------------------------------- | ---------------------------------------- |
| **Experiment Data** | **[Experiment Data Management](experiment_data_management.md)** | How messaging and tracking work together |
| **Messaging**       | **[Messaging System README](../src/messaging/README.md)**       | Real-time communication system           |
| **Tracking**        | **[Tracking System README](../src/tracking/README.md)**         | Persistent experiment logging            |
| **Environment**     | **[Dynamic Environment](dynamic_environment.md)**               | Model-agnostic RL environment            |

### Implementation & Planning

| Document                                          | Purpose                                    | Status    |
| ------------------------------------------------- | ------------------------------------------ | --------- |
| **[Implementation Plan](implementation_plan.md)** | Development roadmap and completed features | Reference |

## 🎯 Documentation by Use Case

### I want to...

**🔧 Set up the platform and run my first experiment**
→ Start with [Getting Started](getting_started.md)

**🏗️ Understand the system architecture**  
→ Read [Architecture](architecture.md) then [Technical Overview](technical_overview.md)

**💻 Contribute to development**
→ Follow [Development Guide](development.md)

**📊 Monitor experiments in real-time**
→ Learn about [Messaging System](../src/messaging/README.md)

**📈 Analyze and compare experiments**
→ Learn about [Tracking System](../src/tracking/README.md)

**🔄 Understand data flow between systems**
→ Read [Experiment Data Management](experiment_data_management.md)

**🎮 Customize the trading environment**
→ Study [Dynamic Environment](dynamic_environment.md)

## 🧩 System Overview

The Agentic Trading platform consists of several key systems:

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENTIC TRADING PLATFORM                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📊 Market Data     🤖 RL Environment     🧠 Agents             │
│     (Qlib + APIs)      (Dynamic)           (Stable-Baselines3)  │
│                                                                 │
│  💬 Messaging       📈 Tracking           🎛️ Dashboard          │
│     (Real-time)        (Persistent)        (Web UI)            │
│                                                                 │
│  🔧 CLI Tools       📋 Configuration     🧪 Testing             │
│     (Scripts)          (YAML)              (Comprehensive)      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Characteristics

-   **📱 Real-time Monitoring**: Live updates during training via messaging system
-   **📊 Rich Analysis**: Comprehensive experiment tracking with Aim backend
-   **🔄 Model Agnostic**: Works with any ML framework or input type
-   **⚙️ Highly Configurable**: YAML-based configuration for all components
-   **🧪 Thoroughly Tested**: 80%+ test coverage with comprehensive test suites
-   **🔧 Production Ready**: Designed for both development and production workflows

## 📖 Reading Order Recommendations

### For New Users

1. [Getting Started](getting_started.md) - Essential first step
2. [Architecture](architecture.md) - Understand the design philosophy
3. [Experiment Data Management](experiment_data_management.md) - Learn the data flow
4. [Dynamic Environment](dynamic_environment.md) - Understand the RL environment

### For Developers

1. [Architecture](architecture.md) - Design principles
2. [Technical Overview](technical_overview.md) - Detailed architecture
3. [Development Guide](development.md) - Development workflows
4. [Messaging System README](../src/messaging/README.md) - Real-time system
5. [Tracking System README](../src/tracking/README.md) - Persistent system

### For System Administrators

1. [Getting Started](getting_started.md) - Installation and setup
2. [Technical Overview](technical_overview.md) - System components
3. [Development Guide](development.md) - Testing and maintenance

## 🆘 Getting Help

### Documentation Issues

-   **Missing information?** Check if it's covered in system-specific READMEs
-   **Unclear instructions?** Look for troubleshooting sections in each guide
-   **Need examples?** Most guides include real-world usage examples

### Common Questions

-   **How do I monitor training?** → [Experiment Data Management](experiment_data_management.md)
-   **How do I compare experiments?** → [Tracking System README](../src/tracking/README.md)
-   **How do I customize the environment?** → [Dynamic Environment](dynamic_environment.md)
-   **How do I contribute?** → [Development Guide](development.md)

### Additional Resources

-   **Tests**: Look at `src/*/tests/` for usage examples
-   **Integration Tests**: Check `integration_tests/` for end-to-end examples
-   **Configuration**: Examine `configs/` directory for setup examples

---

**💡 Tip**: Each system has its own detailed README in its source directory (`src/*/README.md`). These provide the most comprehensive and up-to-date information for specific components.
