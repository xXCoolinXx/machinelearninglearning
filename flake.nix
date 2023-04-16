{
  description = "Machine Learning Learning";
  
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs}: 
  let
    # System types to support.
    supportedSystems = [ "x86_64-linux" "x86_64-darwin" ];

    # Helper function to generate an attrset '{ x86_64-linux = f "x86_64-linux"; ... }'.
    forAllSystems = f:
      nixpkgs.lib.genAttrs supportedSystems (system: f system);
    
    # Nixpkgs instantiated for supported system types.
    nixpkgsFor = forAllSystems (system:
      import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        #overlays = [ self.overlay ];
        config.android_sdk.accept_license = true;
      }
    );
  in 
  {
    #Nix build stuff
    packages = forAllSystems (system:
    let pkgs = nixpkgsFor."${system}";
    in rec {
      ml = pkgs.ml;
    });
    
    #Create a shell environment with the correct libraries
    devShell = forAllSystems (system:
    let pkgs = nixpkgsFor."${system}";
    in pkgs.mkShell {
      buildInputs = with pkgs; [ 
        python310
        python310Packages.tensorflow-bin
        python310Packages.keras
        python310Packages.pillow
      ];
    });
  };
}
