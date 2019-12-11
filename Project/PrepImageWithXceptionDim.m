    function Iout = PrepImageWithXceptionDim(filename)  
                  
        I = imread(filename);  
        if ismatrix(I)  
            I = cat(3,I,I,I);  
        end  
        %% image resizing with xception layers dimensions
        Iout = imresize(I, [299 299]);              
    end  