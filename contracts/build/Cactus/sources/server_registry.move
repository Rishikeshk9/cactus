module cactus::server_registry {
    use std::error;
    use std::signer;
    use std::string::{Self, String};
    use std::vector;
    use aptos_framework::event;
    use aptos_framework::account;

    /// Errors
    const EINVALID_SERVER: u64 = 1;
    const EINVALID_IP_PORT: u64 = 2;
    const EINVALID_PROTOCOL: u64 = 3;
    const EALREADY_REGISTERED: u64 = 5;
    const ENOT_REGISTERED: u64 = 6;
    const EUNAUTHORIZED: u64 = 7;

    /// Server struct to store server information
    struct Server has store, drop, copy {
        ip_port: String,
        protocol: String,
        owner: address
    }

    /// Server registry to track registered servers
    struct ServerRegistry has key {
        servers: vector<Server>,
        register_events: event::EventHandle<ServerRegisteredEvent>,
        update_events: event::EventHandle<ServerUpdatedEvent>,
        remove_events: event::EventHandle<ServerRemovedEvent>,
        admin: address
    }

    /// Events
    #[event]
    struct ServerRegisteredEvent has drop, store {
        ip_port: String,
        protocol: String,
        owner: address
    }

    #[event]
    struct ServerUpdatedEvent has drop, store {
        ip_port: String,
        protocol: String,
        owner: address
    }

    #[event]
    struct ServerRemovedEvent has drop, store {
        owner: address
    }

    /// Initialize the module
    fun init_module(account: &signer) {
        initialize(account);
    }

    #[test_only]
    public fun initialize(account: &signer) {
        let admin_address = signer::address_of(account);
        move_to(account, ServerRegistry {
            servers: vector::empty(),
            register_events: account::new_event_handle<ServerRegisteredEvent>(account),
            update_events: account::new_event_handle<ServerUpdatedEvent>(account),
            remove_events: account::new_event_handle<ServerRemovedEvent>(account),
            admin: admin_address
        });
    }

    /// Register a server
    public fun register_server(
        account: &signer,
        ip_port: String,
        protocol: String
    ) acquires ServerRegistry {
        // Validate inputs
        assert!(!string::is_empty(&ip_port), error::invalid_argument(EINVALID_IP_PORT));
        assert!(!string::is_empty(&protocol), error::invalid_argument(EINVALID_PROTOCOL));

        let owner_address = signer::address_of(account);
        
        // Check if server is already registered
        {
            let registry = borrow_global<ServerRegistry>(@cactus);
            let (found, _) = find_server_index(&registry.servers, owner_address);
            assert!(!found, EALREADY_REGISTERED);
        };

        // Create and add new server
        let registry = borrow_global_mut<ServerRegistry>(@cactus);
        let server = Server {
            ip_port,
            protocol,
            owner: owner_address
        };
        vector::push_back(&mut registry.servers, server);

        // Emit registration event
        event::emit_event(
            &mut registry.register_events,
            ServerRegisteredEvent {
                ip_port,
                protocol,
                owner: owner_address
            }
        );
    }

    /// Update a server's information
    public fun update_server(
        account: &signer,
        ip_port: String,
        protocol: String,
        owner_address: address
    ) acquires ServerRegistry {
        // Validate inputs
        assert!(!string::is_empty(&ip_port), error::invalid_argument(EINVALID_IP_PORT));
        assert!(!string::is_empty(&protocol), error::invalid_argument(EINVALID_PROTOCOL));

        let registry = borrow_global_mut<ServerRegistry>(@cactus);
        let caller_address = signer::address_of(account);
        
        // Find and update server
        let (found, index) = find_server_index(&registry.servers, owner_address);
        assert!(found, ENOT_REGISTERED);

        let server = vector::borrow_mut(&mut registry.servers, index);
        
        // Check authorization - only owner or admin can update
        assert!(
            server.owner == caller_address || registry.admin == caller_address,
            EUNAUTHORIZED
        );
        
        server.ip_port = ip_port;
        server.protocol = protocol;

        // Emit update event
        event::emit_event(
            &mut registry.update_events,
            ServerUpdatedEvent {
                ip_port,
                protocol,
                owner: server.owner
            }
        );
    }

    /// Remove a server from the registry
    public fun remove_server(
        account: &signer,
        owner_address: address
    ) acquires ServerRegistry {
        let registry = borrow_global_mut<ServerRegistry>(@cactus);
        let caller_address = signer::address_of(account);
        
        // Find and remove server
        let (found, index) = find_server_index(&registry.servers, owner_address);
        assert!(found, ENOT_REGISTERED);

        let server = vector::borrow(&registry.servers, index);
        
        // Check authorization - only owner or admin can remove
        assert!(
            server.owner == caller_address || registry.admin == caller_address,
            EUNAUTHORIZED
        );
        
        let owner = server.owner;
        vector::remove(&mut registry.servers, index);

        // Emit removal event
        event::emit_event(
            &mut registry.remove_events,
            ServerRemovedEvent {
                owner
            }
        );
    }

    /// Get server information by owner address
    public fun get_server_by_owner(owner_address: address): Server acquires ServerRegistry {
        let registry = borrow_global<ServerRegistry>(@cactus);
        let (found, index) = find_server_index(&registry.servers, owner_address);
        assert!(found, ENOT_REGISTERED);
        *vector::borrow(&registry.servers, index)
    }

    /// Get all registered servers
    public fun get_all_servers(): vector<Server> acquires ServerRegistry {
        let registry = borrow_global<ServerRegistry>(@cactus);
        registry.servers
    }

    /// Check if a server is registered
    public fun is_server_registered(owner_address: address): bool acquires ServerRegistry {
        let registry = borrow_global<ServerRegistry>(@cactus);
        let (found, _) = find_server_index(&registry.servers, owner_address);
        found
    }

    /// Check if an address is the admin
    public fun is_admin(address: address): bool acquires ServerRegistry {
        let registry = borrow_global<ServerRegistry>(@cactus);
        registry.admin == address
    }

    // Helper function to find server index by owner address
    fun find_server_index(servers: &vector<Server>, owner_address: address): (bool, u64) {
        let i = 0;
        let len = vector::length(servers);
        while (i < len) {
            let server = vector::borrow(servers, i);
            if (server.owner == owner_address) {
                return (true, i)
            };
            i = i + 1;
        };
        (false, 0)
    }

    // Getter functions for Server struct fields
    public fun get_server_ip_port(server: &Server): String {
        server.ip_port
    }

    public fun get_server_protocol(server: &Server): String {
        server.protocol
    }

    public fun get_server_owner_address(server: &Server): address {
        server.owner
    }
} 