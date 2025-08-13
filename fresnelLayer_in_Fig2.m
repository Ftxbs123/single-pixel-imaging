%Fresnel diffraction process
classdef fresnelLayer < nnet.layer.Layer %#codegen
   
    properties
        % imageConv block.
        Network

    end
    
    methods
        function layer = fresnelLayer(inputSize,outputSize,Qr,Qi,NameValueArgs)
            
            % Parse input arguments.
            arguments
                inputSize
                outputSize
                Qr
                Qi
                
                NameValueArgs.Name = ''
            end
             name =NameValueArgs.Name;

            % Set number of inputs.
            layer.NumInputs = 1;
            
            % Set layer name.
            layer.Name = name;
      
            % Set layer description.
            layer.Description = "Fresnel proagation of phase-only input";
           
            % Define nested layer graph.
            lgraph = layerGraph;
            layers = [
                imageInputLayer([inputSize 1],'Normalization','None','Name','in')];
%                 plusLayer(P,'plus')
            lgraph = addLayers(lgraph,layers);
                     
            lgraph = addLayers(lgraph,cosLayer('cos'));
            lgraph = addLayers(lgraph,sinLayer('sin'));
            lgraph = connectLayers(lgraph,'in','cos');
            lgraph = connectLayers(lgraph,'in','sin');
            
            % zero padding
            lgraph = addLayers(lgraph,ZeroPadding2dLayer('cospad', (outputSize - inputSize)/2));
            lgraph = addLayers(lgraph,ZeroPadding2dLayer('sinpad', (outputSize - inputSize)/2));
            lgraph = connectLayers(lgraph,'cos','cospad');
            lgraph = connectLayers(lgraph,'sin','sinpad');
            
            % fft2
            lgraph = addLayers(lgraph,fft2DLayer('Fcos'));
            lgraph = addLayers(lgraph,fft2DLayer('Fsin'));
            lgraph = connectLayers(lgraph,'cospad','Fcos');
            lgraph = connectLayers(lgraph,'sinpad','Fsin');
            
            lgraph = addLayers(lgraph,subtractionLayer('Fr'));
            lgraph = addLayers(lgraph,additionLayer(2,'Name','Fi'));
            lgraph = connectLayers(lgraph,'Fcos/real','Fr/in1');
            lgraph = connectLayers(lgraph,'Fsin/imag','Fr/in2');
            lgraph = connectLayers(lgraph,'Fcos/imag','Fi/in1');
            lgraph = connectLayers(lgraph,'Fsin/real','Fi/in2');

            % fftshift
            lgraph = addLayers(lgraph,fftshiftLayer('Frs'));
            lgraph = addLayers(lgraph,fftshiftLayer('Fis'));
            lgraph = connectLayers(lgraph,'Fr','Frs');
            lgraph = connectLayers(lgraph,'Fi','Fis');

            %Q
            lgraph = addLayers(lgraph,mulLayer(Qr,'FrQr')); 
            lgraph = addLayers(lgraph,mulLayer(Qi,'FiQi'));
            lgraph = addLayers(lgraph,mulLayer(Qi,'FrQi'));
            lgraph = addLayers(lgraph,mulLayer(Qr,'FiQr'));
            lgraph = connectLayers(lgraph,'Frs','FrQr');
            lgraph = connectLayers(lgraph,'Fis','FiQi');
            lgraph = connectLayers(lgraph,'Frs','FrQi');
            lgraph = connectLayers(lgraph,'Fis','FiQr');
            lgraph = addLayers(lgraph,subtractionLayer('Frn'));
            lgraph = connectLayers(lgraph,'FrQr','Frn/in1');
            lgraph = connectLayers(lgraph,'FiQi','Frn/in2');
            lgraph = addLayers(lgraph,additionLayer(2,'Name','Fin'));
            lgraph = connectLayers(lgraph,'FrQi','Fin/in1');
            lgraph = connectLayers(lgraph,'FiQr','Fin/in2');
            % ifftshift
            lgraph = addLayers(lgraph,ifftshiftLayer('Frnishift'));
            lgraph = addLayers(lgraph,ifftshiftLayer('Finishift'));
            lgraph = connectLayers(lgraph,'Frn','Frnishift');
            lgraph = connectLayers(lgraph,'Fin','Finishift');
            % ifft2
            lgraph = addLayers(lgraph,ifft2DLayer('iFrn'));
            lgraph = addLayers(lgraph,ifft2DLayer('iFin'));
            lgraph = connectLayers(lgraph,'Frnishift','iFrn');
            lgraph = connectLayers(lgraph,'Finishift','iFin');
            
            lgraph = addLayers(lgraph,subtractionLayer('Frnn'));
            lgraph = addLayers(lgraph,additionLayer(2,'Name','Finn'));
            lgraph = connectLayers(lgraph,'iFrn/real','Frnn/in1');
            lgraph = connectLayers(lgraph,'iFin/imag','Frnn/in2');
            lgraph = connectLayers(lgraph,'iFrn/imag','Finn/in1');
            lgraph = connectLayers(lgraph,'iFin/real','Finn/in2');

            % intensity
            lgraph = addLayers(lgraph,intensityLayer('I'));
            lgraph = connectLayers(lgraph,'Frnn','I/in1');
            lgraph = connectLayers(lgraph,'Finn','I/in2');
            
            
            
            % Convert to dlnetwork.
            dlnet = dlnetwork(lgraph);
    
            % Set Network property.
            layer.Network = dlnet;
            
        end
        
        function Z = predict(layer, X)
            X = dlarray(X,'SSC');
            dlnet = layer.Network;
            Z = predict(dlnet,X);
            Z = stripdims(Z);
            
        end

    end
end